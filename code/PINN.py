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
        header = "Epoch,Time,LR,Loss,Loss_data,Loss_T_bound,Loss_NSE,Loss_EE,Loss_conti,MAE_u,MAE_v,MAE_w,MAE_T,MAE_p,COR_u,COR_v,COR_w,COR_T,COR_p\n"
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
        u_true, v_true, w_true, T_true, p_true = tf.unstack(y_true, axis=1)
        u, v, w, T, p = tf.unstack(y_pred, axis=1)
        # estimate evaluation metrics
        mae = { 'MAE_u': keras.losses.mean_absolute_error(u_true, u),
                'MAE_v': keras.losses.mean_absolute_error(v_true, v),
                'MAE_w': keras.losses.mean_absolute_error(w_true, w),
                'MAE_T': keras.losses.mean_absolute_error(T_true, T),
                'MAE_p': keras.losses.mean_absolute_error(p_true, p)}
        cor = { 'COR_u': pearson_correlation(u_true, u),
                'COR_v': pearson_correlation(v_true, v),
                'COR_w': pearson_correlation(w_true, w),
                'COR_T': pearson_correlation(T_true, T),
                'COR_p': pearson_correlation(p_true, p)}
        # write output
        l1, l2, l3, l4, l5, l6 = logs['loss'], logs['loss_data'], logs['loss_T_bound'], logs['loss_NSE'], logs['loss_EE'], logs['loss_conti']
        l7 = logs['learning_rate']
        eu, ev, ew, eT, ep = float(mae['MAE_u'].numpy()), float(mae['MAE_v'].numpy()), float(mae['MAE_w'].numpy()), float(mae['MAE_T'].numpy()), float(mae['MAE_p'].numpy())
        cu, cv, cw, cT, cp = float(cor['COR_u'].numpy()), float(cor['COR_v'].numpy()), float(cor['COR_w'].numpy()), float(cor['COR_T'].numpy()), float(cor['COR_p'].numpy())
        epoch_time = time.time() - self.epoch_start_time
        # Prepare log string
        log_str = f"{epoch+1},{epoch_time:.2f},{l7:.6f},{l1:.7f},{l2:.7f},{l3:.7f},{l4:.7f},{l5:.7f},{l6:.7f},"
        log_str += f"{eu:.7f},{ev:.7f},{ew:.7f},{eT:.7f},{ep:.7f},{cu:.7f},{cv:.7f},{cw:.7f},{cT:.7f},{cp:.7f}\n"
        # Write to log file
        self.log_file.write(log_str)
        self.log_file.flush()  # Ensure the data is written immediately
        # Print to console
        print(f"Epoch: {epoch+1}/{self.N_epochs} - Time: {epoch_time:.2f}s - LR: {l7:.6f}")
        print(f"  loss: {l1:.7f} - loss_data: {l2:.7f} - loss_T_bound: {l3:.7f} - loss_NSE: {l4:.7f} - loss_EE: {l5:.7f} - loss_conti: {l6:.7f}")
        print(f"  mae_u: {eu:.7f} - mae_v: {ev:.7f} - mae_w: {ew:.7f} - mae_T: {eT:.7f} - mae_p: {ep:.7f}")
        print(f"  cor_u: {cu:.7f} - cor_v: {cv:.7f} - cor_w: {cw:.7f} - cor_T: {cT:.7f} - cor_p: {cp:.7f}\n")
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
        model.add(layers.Dense(5))   # Output is u,v,w,T,p
        return model
    
    def call(self, inputs):
        y_pred = self.model(inputs)
        return y_pred

    def loss_function(self, inputs, y_true):
        t, x, y, z = tf.unstack(inputs, axis=1)
        u_true, v_true, w_true, T_true, p_true = tf.unstack(y_true, axis=1)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            # Predict the fields
            y_pred = self(inputs, training=True)
            u, v, w, T, p = tf.unstack(y_pred, axis=1)
                
            # first derivatives
            u_t, u_x, u_y, u_z = tf.unstack(tape.gradient(u, inputs), axis=-1)
            v_t, v_x, v_y, v_z = tf.unstack(tape.gradient(v, inputs), axis=-1)
            w_t, w_x, w_y, w_z = tf.unstack(tape.gradient(w, inputs), axis=-1)
            T_t, T_x, T_y, T_z = tf.unstack(tape.gradient(T, inputs), axis=-1)
            p_t, p_x, p_y, p_z = tf.unstack(tape.gradient(p, inputs), axis=-1)

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
            
            T_xx = tape.gradient(T_x, inputs)[...,1]
            T_yy = tape.gradient(T_y, inputs)[...,2]
            T_zz = tape.gradient(T_z, inputs)[...,3]          
            
            del tape
            
        # --------------------------------
        # data loss: calculate the mean squared error for  u,v,w
        # --------------------------------
        if self.pmode:
            loss_data = keras.losses.mean_squared_error(tf.concat([u_true, v_true, w_true, p_true], axis=-1), tf.concat([u, v, w, p], axis=-1))
        else:
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
        NSE_u = u_t + u*u_x + v*u_y + w*u_z + p_x - np.sqrt(self.Pr/self.Ra) * (u_xx + u_yy + u_zz)
        NSE_v = v_t + u*v_x + v*v_y + w*v_z + p_y - np.sqrt(self.Pr/self.Ra) * (v_xx + v_yy + v_zz)
        NSE_w = w_t + u*w_x + v*w_y + w*w_z + p_z - np.sqrt(self.Pr/self.Ra) * (w_xx + w_yy + w_zz) - T
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