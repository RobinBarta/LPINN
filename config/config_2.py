class Parameter:
    # ------------------
    # data parameters
    # ------------------
    data_path = 'data/RBC_exp_2e9_7_t_51.txt'
    load_initial = False 
    initial_path = ''
    initial_weight = 0
    
    # ------------------
    # PINN parameters
    # ------------------
    N_layer = 8
    N_neuron = 128
    batch_size = 4096
    epochs = 20000
    learning_rate = 1e-3
    
    # ---------------------
    # physics parameters
    # ---------------------
    Ra = 2e9
    Pr = 7.0
    
    # ----------
    # weights
    # ----------
    lambda_data = 1.0
    lambda_T_bound = 1e-3
    lambda_conti = 1e-3
    lambda_NSE = 0.1
    lambda_EE = 1e-2