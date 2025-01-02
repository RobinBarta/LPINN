class Parameter:
    # ------------------
    # data parameters
    # ------------------
    data_path = 'data/RBC_exp_2E9_7_t_42.txt'
    pmode = False
    Ra = 2e9
    Pr = 7.0
    
    # ------------------
    # initialization
    # ------------------
    load_initial = True 
    initial_path = 'vor'
    initial_weight = 3000
    
    # ------------------
    # PINN parameters
    # ------------------
    N_layer = 8
    N_neuron = 128
    batch_size = 4096
    epochs = 3000
    learning_rate, min_lr = 1e-3, 1e-5
    reduction_epochs, reduction_factor = 300, 0.5
    
    # ----------
    # weights
    # ----------
    lambda_data = 1.0
    lambda_T_bound = 1e-3
    lambda_conti = 1e-3
    lambda_NSE = 0.1
    lambda_EE = 1e-3