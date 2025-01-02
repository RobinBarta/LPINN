class Parameter:
    # ------------------
    # data parameters
    # ------------------
    data_path = 'data/RBC_dns_1E10_7_t_42.txt'
    pmode = True
    Ra = 1e10
    Pr = 7.0
    
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
    epochs = 500
    learning_rate, min_lr = 1e-3, 1e-5
    reduction_epochs, reduction_factor = 60, 0.5
    
    # ----------
    # weights
    # ----------
    lambda_data = 1.0
    lambda_T_bound = 1e-3
    lambda_conti = 1e-3
    lambda_NSE = 0.1
    lambda_EE = 5e-3