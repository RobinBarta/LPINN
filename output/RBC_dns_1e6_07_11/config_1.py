class Parameter:
    # ------------------
    # data parameters
    # ------------------
    data_path = 'data/RBC_dns_1E6_07_t_11.txt'
    pmode = False
    Ra = 1e6
    Pr = 0.7
    
    # ------------------
    # initialization
    # ------------------
    load_initial = False 
    initial_path = ''
    initial_weight = 0
    
    # ------------------
    # PINN parameters
    # ------------------
    N_layer = 8
    N_neuron = 256
    batch_size = 4096
    epochs = 1000
    learning_rate, min_lr = 1e-3, 1e-5
    reduction_epochs, reduction_factor = 60, 0.5
    
    # ----------
    # weights
    # ----------
    lambda_data = 1.0
    lambda_T_bound = 1e-3
    lambda_conti = 1e-3
    lambda_NSE = 0.1
    lambda_EE = 1e-2