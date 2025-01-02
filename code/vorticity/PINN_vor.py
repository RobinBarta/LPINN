# %% used libraries
import os, logging, time
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import keras

from keras import layers
# %%


def pearson_correlation(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x_mean = K.mean(x)
    y_mean = K.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    # Calculate the numerator as the sum of the product of centered data points
    numerator = K.sum(x_centered * y_centered)
    # Calculate the denominator as the product of the standard deviations
    denominator = K.sqrt(K.sum(K.square(x_centered)) * K.sum(K.square(y_centered)))   
    # Calculate the Pearson correlation coefficient
    pearson_corr = numerator / (denominator + K.epsilon())  # Adding epsilon to avoid division by zero
    return pearson_corr


class CustomLoggingCallback(keras.callbacks.Callback):
    def __init__(self, test_data, model, N_epochs, log_dir, param):
        super(CustomLoggingCallback, self).__init__()
        self.test_data = test_data  # Store test dataset
        self._model = model         # Use a protected attribute for the model
        self.N_epochs = N_epochs
        self.epoch_start_time = None 
        self.log_dir = log_dir
        self.param = param
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # Create or open the log file
        self.log_file = open(os.path.join(log_dir, 'training_log.txt'), 'a')
        # Write header to the log file
        header = "Epoch,Time,LR,Loss,Loss_data,Loss_T_bound,Loss_NSE,Loss_EE,Loss_conti,MAE_u,MAE_v,MAE_w,MAE_T,COR_u,COR_v,COR_w,COR_T\n"
        self.log_file.write(header)
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, value):
        self._model = value
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
    def on_epoch_end(self, epoch, logs=None):
        inputs, y_true = self.test_data
        y_pred = self.model(inputs, training=False)
        # unstack output and ground truth
        u_true, v_true, w_true, T_true = tf.unstack(y_true, axis=1)
        u, v, w, T = tf.unstack(y_pred, axis=1)
        # estimate evaluation metrics
        mae = { 'MAE_u': keras.losses.mean_absolute_error(u_true, u),
                'MAE_v': keras.losses.mean_absolute_error(v_true, v),
                'MAE_w': keras.losses.mean_absolute_error(w_true, w),
                'MAE_T': keras.losses.mean_absolute_error(T_true, T)}
        cor = { 'COR_u': pearson_correlation(u_true, u),
                'COR_v': pearson_correlation(v_true, v),
                'COR_w': pearson_correlation(w_true, w),
                'COR_T': pearson_correlation(T_true, T)}
        # write output
        l1, l2, l3, l4, l5, l6 = logs['loss'], logs['loss_data'], logs['loss_T_bound'], logs['loss_NSE'], logs['loss_EE'], logs['loss_conti']
        l7 = logs['learning_rate']
        eu, ev, ew, eT = float(mae['MAE_u'].numpy()), float(mae['MAE_v'].numpy()), float(mae['MAE_w'].numpy()), float(mae['MAE_T'].numpy())
        cu, cv, cw, cT = float(cor['COR_u'].numpy()), float(cor['COR_v'].numpy()), float(cor['COR_w'].numpy()), float(cor['COR_T'].numpy())
        epoch_time = time.time() - self.epoch_start_time
        # Prepare log string
        log_str = f"{epoch+1},{epoch_time:.2f},{l7:.6f},{l1:.7f},{l2:.7f},{l3:.7f},{l4:.7f},{l5:.7f},{l6:.7f},"
        log_str += f"{eu:.7f},{ev:.7f},{ew:.7f},{eT:.7f},{cu:.7f},{cv:.7f},{cw:.7f},{cT:.7f}\n"
        # Write to log file
        self.log_file.write(log_str)
        self.log_file.flush()  # Ensure the data is written immediately
        # Print to console
        print(f"Epoch: {epoch+1}/{self.N_epochs} - Time: {epoch_time:.2f}s - LR: {l7:.6f}")
        print(f"  loss: {l1:.7f} - loss_data: {l2:.7f} - loss_T_bound: {l3:.7f} - loss_NSE: {l4:.7f} - loss_EE: {l5:.7f} - loss_conti: {l6:.7f}")
        print(f"  mae_u: {eu:.7f} - mae_v: {ev:.7f} - mae_w: {ew:.7f} - mae_T: {eT:.7f}")
        print(f"  cor_u: {cu:.7f} - cor_v: {cv:.7f} - cor_w: {cw:.7f} - cor_T: {cT:.7f}\n")
        # do lr reduction
        if (epoch + 1) % self.param.reduction_epochs == 0:
            current_lr = self.model.optimizer.learning_rate.numpy()
            new_lr = max(current_lr * self.param.reduction_factor, self.param.min_lr)
            self.model.optimizer.learning_rate.assign(new_lr)
    def on_train_end(self, logs=None):
        # Close the log file when training ends
        self.log_file.close()


class PINN(keras.Model):
    def __init__(self, params):
        super(PINN, self).__init__() # Pass kwargs to the parent class
        self.params = params
        # network architecture
        self.N_layer = self.params.N_layer
        self.N_neuron = self.params.N_neuron
        # dimensionless numbers
        self.Pr = np.float32(self.params.Pr)
        self.Ra = np.float32(self.params.Ra)
        # weights
        self.lambda_data = self.params.lambda_data
        self.lambda_T_bound = self.params.lambda_T_bound
        self.lambda_conti = self.params.lambda_conti
        self.lambda_NSE = self.params.lambda_NSE
        self.lambda_EE = self.params.lambda_EE
        # training mode
        self.pmode = self.params.pmode
        
        self.activation_function = lambda a: 0.5 * tf.sin(2*np.pi*a)
        self.model = self.build_model()
        
    def build_model(self):
        model = keras.Sequential()
        # input layer
        model.add(layers.Input(shape=(4,)))  # 4 inputs: t, x, y, z
        # hidden layers
        for i in range(self.N_layer):
            model.add(layers.Dense(self.N_neuron, activation=self.activation_function))
        # output layer
        model.add(layers.Dense(4))   # Output is u,v,w,T
        return model
    
    def call(self, inputs):
        y_pred = self.model(inputs)
        return y_pred

    def loss_function(self, inputs, y_true):
        t, x, y, z = tf.unstack(inputs, axis=1)
        u_true, v_true, w_true, T_true = tf.unstack(y_true, axis=1)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            # Predict the fields
            y_pred = self(inputs, training=True)
            u, v, w, T = tf.unstack(y_pred, axis=1)
                
            # first derivatives
            u_t, u_x, u_y, u_z = tf.unstack(tape.gradient(u, inputs), axis=-1)
            v_t, v_x, v_y, v_z = tf.unstack(tape.gradient(v, inputs), axis=-1)
            w_t, w_x, w_y, w_z = tf.unstack(tape.gradient(w, inputs), axis=-1)
            T_t, T_x, T_y, T_z = tf.unstack(tape.gradient(T, inputs), axis=-1)

            # second derivatives
            u_xx = tape.gradient(u_x, inputs)[...,1]
            u_yy = tape.gradient(u_y, inputs)[...,2]
            u_zz = tape.gradient(u_z, inputs)[...,3]
            
            v_xx = tape.gradient(v_x, inputs)[...,1]
            v_yy = tape.gradient(v_y, inputs)[...,2]
            v_zz = tape.gradient(v_z, inputs)[...,3]
            
            w_xx = tape.gradient(w_x, inputs)[...,1]
            w_yy = tape.gradient(w_y, inputs)[...,2]
            w_zz = tape.gradient(w_z, inputs)[...,3]
            
            u_xt, u_xx, u_xy, u_xz = tf.unstack(tape.gradient(u_x, inputs), axis=-1)
            v_xt, v_xx, v_xy, v_xz = tf.unstack(tape.gradient(v_x, inputs), axis=-1)
            w_xt, w_xx, w_xy, w_xz = tf.unstack(tape.gradient(w_x, inputs), axis=-1)
            
            u_yt, u_yx, u_yy, u_yz = tf.unstack(tape.gradient(u_y, inputs), axis=-1)
            v_yt, v_yx, v_yy, v_yz = tf.unstack(tape.gradient(v_y, inputs), axis=-1)
            w_yt, w_yx, w_yy, w_yz = tf.unstack(tape.gradient(w_y, inputs), axis=-1)
            
            u_zt, u_zx, u_zy, u_zz = tf.unstack(tape.gradient(u_z, inputs), axis=-1)
            v_zt, v_zx, v_zy, v_zz = tf.unstack(tape.gradient(v_z, inputs), axis=-1)
            w_zt, w_zx, w_zy, w_zz = tf.unstack(tape.gradient(w_z, inputs), axis=-1)
            
            T_xx = tape.gradient(T_x, inputs)[...,1]
            T_yy = tape.gradient(T_y, inputs)[...,2]
            T_zz = tape.gradient(T_z, inputs)[...,3] 
            
            # third derivatives
            u_yxx = tape.gradient(u_yx, inputs)[...,1]
            u_zxx = tape.gradient(u_zx, inputs)[...,1]
            u_yyy = tape.gradient(u_yy, inputs)[...,2]
            u_zyy = tape.gradient(u_zy, inputs)[...,2]
            u_yzz = tape.gradient(u_yz, inputs)[...,3]
            u_zzz = tape.gradient(u_zz, inputs)[...,3]
            
            v_xxx = tape.gradient(v_xx, inputs)[...,1]
            v_zxx = tape.gradient(v_zx, inputs)[...,1]
            v_xyy = tape.gradient(v_xy, inputs)[...,2]
            v_zyy = tape.gradient(v_zy, inputs)[...,2]
            v_xzz = tape.gradient(v_xz, inputs)[...,3]
            v_zzz = tape.gradient(v_zz, inputs)[...,3]
            
            w_xxx = tape.gradient(w_xx, inputs)[...,1]
            w_yxx = tape.gradient(w_yx, inputs)[...,1]
            w_xyy = tape.gradient(w_xy, inputs)[...,2]
            w_yyy = tape.gradient(w_yy, inputs)[...,2]
            w_xzz = tape.gradient(w_xz, inputs)[...,3]
            w_yzz = tape.gradient(w_yz, inputs)[...,3]
            
            del tape
            
        # --------------------------------
        # data loss: calculate the mean squared error for  u,v,w
        # --------------------------------
        loss_data = keras.losses.mean_squared_error(tf.concat([u_true, v_true, w_true], axis=-1), tf.concat([u, v, w], axis=-1))
        
        # --------------------------------
        # boundary loss for Temperature
        # --------------------------------
        z0, z1 = 0.0, 1.0
        T_min, T_max = -0.5, 0.5
        N = tf.shape(inputs)[0]
        t_reshaped = tf.reshape(t, (N,1))
        xy_bound = tf.random.stateless_uniform((N,2), seed=[22,4])
        z_bound = tf.cast(tf.random.stateless_uniform(shape=(N, 1),seed=[22,4], minval=0, maxval=2, dtype=tf.int32), tf.float32)
        inputs_bound = tf.concat([t_reshaped, xy_bound, z_bound], axis=1)
        y_pred_bound = self(inputs_bound, training=False)
        T_bound = y_pred_bound[...,3]
        T_true_bound = tf.reshape(tf.where(z_bound == z0, T_max, tf.where(z_bound == z1, T_min, z_bound)), (N,))
        loss_T_bound = keras.losses.mean_squared_error(T_true_bound, T_bound)
        
        # --------------------------------
        # loss continuity equation
        # --------------------------------
        divU = u_x + v_y + w_z
        loss_conti = tf.reduce_mean(tf.square(divU))
        
        # --------------------------------
        # loss NSE
        # --------------------------------
        NSE_u = (w_yt - v_zt) + u*(w_yx-v_zx) + v*(w_yy-v_zy) + w*(w_yz-v_zz) - (w_y-v_z)*u_x - (u_z-w_x)*u_y - (v_x-u_y)*u_z - tf.sqrt(self.Pr/self.Ra)*(w_yxx-v_zxx + w_yyy-v_zyy + w_yzz-v_zzz) - T_y
        NSE_v = (u_zt - w_xt) + u*(u_zx-w_xx) + v*(u_zy-w_xy) + w*(u_zz-w_xz) - (w_y-v_z)*v_x - (u_z-w_x)*v_y - (v_x-u_y)*v_z - tf.sqrt(self.Pr/self.Ra)*(u_zxx-w_xxx + u_zyy-w_xyy + u_zzz-w_xzz) + T_x
        NSE_w = (v_xt - u_yt) + u*(v_xx-u_yx) + v*(v_xy-u_yy) + w*(v_xz-u_yz) - (w_y-v_z)*w_x - (u_z-w_x)*w_y - (v_x-u_y)*w_z - tf.sqrt(self.Pr/self.Ra)*(v_xxx-u_yxx + v_xyy-u_yyy + v_xzz-u_yzz)
        loss_NSE = tf.reduce_mean(tf.square(tf.stack([NSE_u, NSE_v, NSE_w])))
        
        # --------------------------------
        # loss Energy Equation
        # --------------------------------
        EE = T_t + u*T_x + v*T_y + w*T_z - np.sqrt(1/(self.Pr*self.Ra)) * (T_xx + T_yy + T_zz)
        loss_EE = tf.reduce_mean(tf.square(EE))
        
        # combine loss function
        total_loss = self.lambda_data*loss_data + self.lambda_conti*loss_conti + self.lambda_NSE*loss_NSE + self.lambda_EE*loss_EE + self.lambda_T_bound*loss_T_bound 

        # learning rate
        lr = float(self.optimizer.learning_rate)
    
        #return total_loss
        return {'loss': total_loss, 'loss_data': loss_data, 'loss_T_bound': loss_T_bound, 'loss_conti': loss_conti, 'loss_NSE': loss_NSE, 'loss_EE': loss_EE, 'learning_rate': lr}

    @tf.function
    def train_step(self, data):
        inputs, y_true = data
        with tf.GradientTape() as tape:
            loss_dict = self.loss_function(inputs, y_true)
        loss = loss_dict['loss']
        # compute gradient
        gradients = tape.gradient(loss, self.trainable_variables)
        # apply gradient to update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss_dict
    
    @tf.function
    def test_step(self, data):
        inputs, y_true = data
        #y_pred = self(inputs, training=False)
        return {}