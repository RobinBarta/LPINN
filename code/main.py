'''
----------------------------------------------------------------
|    Project     :   PINN for RBC in Boussinesq Approximation   |
|    Author      :   Michael Mommert                            |
|    CoAuthor    :   Robin Barta, Christian Bauer, Marie Volk   |
|    Copyright   :   DLR                                        |
----------------------------------------------------------------

    ∂_t u + (u*𝜵)u = -𝜵p + sqrt(Pr/Ra)*Δu + T*e_z
    ∂_t T + (u*𝜵)T = sqrt(1/Pr/Ra)*ΔT
               𝜵*u = 0
'''
# %% used libraries
import sys, os, datetime, shutil, logging, glob, random, pickle
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import importlib.util
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split

from PINN import *
# %%

# set random seed
tf.random.set_seed(3), np.random.seed(2), random.seed(2204)

# clear all previously registered custom objects
tf.keras.utils.get_custom_objects().clear()

def main(): 
    # ---------------------------------------
    # initialize parameters from config.py
    # ---------------------------------------
    if len(sys.argv) != 2:
        print("Usage: python ./code/main.py config/config.py")
        sys.exit()
    config_path = sys.argv[1]
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    params = config.Parameter()
    
    # ---------------------------------------------------------------------------
    # create output folder, save the current config file, creat weights folder
    # ---------------------------------------------------------------------------
    params.output_path = 'output/PINN_' + config_path[7:-3].replace('/', '_') + '_' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    os.mkdir(params.output_path)
    shutil.copy(config_path, params.output_path)
    os.mkdir(params.output_path + '/weights')
    
    # -----------------------------------
    # load data
    # -----------------------------------
    data = np.loadtxt(params.data_path,skiprows=1)
    data_input, data_output = data[:,0:4], data[:,4:9]
    # preprocessing
    [data_input, data_output] = DataProcessor([data_input, data_output]).process()
    print('Input shape: ', data_input.shape)
    print('Output shape: ', data_output.shape)
    # random train/test split: Use 5% for validation
    inputs, inputs_val, y_true, y_true_val = train_test_split(data_input, data_output, test_size = 0.05)
    
    # -------------------
    # build PINN model
    # -------------------   
    initial_epoch = 0
    pinn = PINN(params)
    # create a dummy input to build the model, this call creates the model's variables
    _ = pinn(tf.zeros((1, 4)))  
    pinn.model.summary()
    if params.load_initial == True:
        print('Transferring weights from initial model.')
        pinn.load_weights('output/'+params.initial_path+'/weights/weights_epoch_'+f"{int(params.initial_weight):04d}"+'.weights.h5')
        initial_epoch = params.initial_weight
    
    # ---------------------
    # compile PINN model
    # ---------------------
    pinn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate))
    
    # -------------------
    # train PINN model
    # -------------------
    # checkpoint for saving weights after each epoch
    save_weights = tf.keras.callbacks.ModelCheckpoint(params.output_path+'/weights/weights_epoch_{epoch:04d}.weights.h5', save_weights_only=True, save_freq='epoch')
    # reduce learning rate when loss has stopped improving
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=1e-5, min_delta=5e-6)
    # custom logging
    custom_log = CustomLoggingCallback(test_data=[inputs_val,y_true_val], model=pinn, N_epochs=(initial_epoch+params.epochs), log_dir=params.output_path+'/logs')
    # train PINN model
    pinn.fit(x=inputs, y=y_true, batch_size=params.batch_size, epochs=initial_epoch+params.epochs, initial_epoch=initial_epoch, validation_data=(inputs_val, y_true_val), verbose=0, callbacks=[custom_log, save_weights, reduce_lr])
if __name__ == "__main__":
    main()