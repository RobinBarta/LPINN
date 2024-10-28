'''

    This script loads the trained PINN and plots predicted fields.

'''


# %% used libraries

import sys, os, pickle, datetime, shutil, cv2, h5py
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import importlib.util
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.keras import layers
from keras import backend as K
from matplotlib.colors import TwoSlopeNorm

from PINN import *

# %%

class PredictionParameter():
    # physical parameter
    Ra, Pr = 1e6, 0.7
    # number where the a plane is cut to plot the intersection
    t_steps = 3
    d, x1 = 2/63, 1.0
    scatter_size = 5
    # path to data
    data_path = 'RBC_dns_1e6_07_t_11.txt'
    # path the the model weights
    model_path, config_name, weight = 'PINN_config_1_2024-10-22_09-01-19', 'config_1', 16000

def main(): 
    # define network with config file
    params = PredictionParameter()
    
    # -----------------------------------
    # load data
    # -----------------------------------
    data = np.loadtxt('../data/'+params.data_path,skiprows=1)
    data_input, data_output = data[:,0:4], data[:,4:9] 
    # preprocessing
    data = DataProcessor([data_input, data_output]).process()
    data_input, data_output = data[0], data[1]
    
    # load config
    spec = importlib.util.spec_from_file_location("config", '../output/'+params.model_path+'/'+params.config_name+'.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    params2 = config.Parameter()
    
    # -------------------
    # build PINN model
    # -------------------   
    pinn = PINN(params2)
    # create a dummy input to build the model, this call creates the model's variables
    _ = pinn(tf.zeros((1, 4)))  
    pinn.load_weights('../output/'+params.model_path+'/weights/weights_epoch_'+f"{int(params.weight):04d}"+'.weights.h5')
        
    # define output folder
    os.makedirs('../output/'+params.model_path+'/prediction', exist_ok = True)
    os.makedirs('../output/'+params.model_path+'/prediction/data', exist_ok = True)
    os.makedirs('../output/'+params.model_path+'/prediction/data/u', exist_ok = True)
    os.makedirs('../output/'+params.model_path+'/prediction/data/v', exist_ok = True)
    os.makedirs('../output/'+params.model_path+'/prediction/data/w', exist_ok = True)
    os.makedirs('../output/'+params.model_path+'/prediction/data/T', exist_ok = True)
    os.makedirs('../output/'+params.model_path+'/prediction/data/p', exist_ok = True)
    
    # predict
    if not os.path.exists('../output/'+params.model_path+'/prediction/data_predicted.txt'):
        prediction = pinn(data_input)
        prediction = np.append(data_input,prediction,axis=1)
        np.savetxt('../output/'+params.model_path+'/prediction/data_predicted.txt', prediction,header='x,y,z,u,v,w,T,p')
    else:
        prediction = np.loadtxt('../output/'+params.model_path+'/prediction/data_predicted.txt',skiprows=1)
        
    # load training history
    log = np.loadtxt('../output/'+params.model_path+'/logs/training_log.txt',delimiter=',',skiprows=1)
    
    # plot losses
    plt.figure(figsize=(15,10))
    plt.semilogy(log[:,2], label='learning_rate')
    plt.semilogy(log[:,3], label='loss')
    plt.semilogy(log[:,4], label='loss_data')
    plt.semilogy(log[:,5], label='loss_T_bound')
    plt.semilogy(log[:,6], label='loss_NSE')
    plt.semilogy(log[:,7], label='loss_EE')
    plt.semilogy(log[:,8], label='loss_conti')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('../output/'+params.model_path+'/prediction/train_loss.png')
    plt.close()
    
    # plot cor
    plt.figure(figsize=(15,10))
    plt.semilogy(log[:,13], label='cor_u',c='black')
    plt.semilogy(log[:,14], label='cor_v',c='red')
    plt.semilogy(log[:,15], label='cor_w',c='green')
    plt.semilogy(log[:,16], label='cor_T',c='blue')
    plt.xlabel('epoch')
    plt.ylabel('correlation')
    plt.legend()
    plt.savefig('../output/'+params.model_path+'/prediction/train_cor.png')
    plt.close()
    
    # plot cor
    plt.figure(figsize=(15,10))
    plt.semilogy(log[:,9], label='mae_u',c='black')
    plt.semilogy(log[:,10], label='mae_v',c='red')
    plt.semilogy(log[:,11], label='mae_w',c='green')
    plt.semilogy(log[:,12], label='mae_T',c='blue')
    plt.xlabel('epoch')
    plt.ylabel('mae')
    plt.legend()
    plt.savefig('../output/'+params.model_path+'/prediction/train_mae.png')
    plt.close()
    
    # plot prediction
    for i, t in enumerate(tqdm(np.unique(data_input[:,0])[:params.t_steps],desc='Prediction: ',position=0,leave=True)):
        # filter by time step
        ID = np.argwhere(data_input==t)[:,0]
        truth = np.append(data_input[ID,1:],data_output[ID,:],axis=1)
        pred = prediction[ID,1:]
        # filter by cell diagonals
        ID_diag1 = np.argwhere(np.abs(truth[:,0]-params.x1/2)<params.d)[:,0]
        #ID_diag1 = np.argwhere(np.abs(truth[:,0]-truth[:,1])<params.d)[:,0]
        #ID_diag2 = np.argwhere(np.abs(truth[:,1]-(params.x1-truth[:,0]))<params.d)[:,0] 
        ID_diag2 = np.argwhere(np.abs(truth[:,1]-params.x1/2)<params.d)[:,0]
        
        # plot U
        c = 3
        fig, ax = plt.subplots(2,2, figsize=(14,14), sharey=True, sharex=True)
        u1 = ax[0,0].scatter(truth[ID_diag1,1], truth[ID_diag1,2], c=truth[ID_diag1,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        u1p = ax[1,0].scatter(pred[ID_diag1,1], pred[ID_diag1,2], c=pred[ID_diag1,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        u2 = ax[0,1].scatter(truth[ID_diag2,0], truth[ID_diag2,2], c=truth[ID_diag2,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        u2p = ax[1,1].scatter(pred[ID_diag2,0], pred[ID_diag2,2], c=pred[ID_diag2,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        ax[0,0].set_title(r'$u$'), ax[0,1].set_title(r'$u$'), ax[1,0].set_title(r'$\tilde{u}$'), ax[1,1].set_title(r'$\tilde{u}$')
        plt.colorbar(u1), plt.colorbar(u1p), plt.colorbar(u2), plt.colorbar(u2p)                
        plt.setp(ax[-1,0],xlabel=r'$X : |x-y|<$'+str(params.d)), plt.setp(ax[-1,1],xlabel=r'$X : |y-(1-x)|<$'+str(params.d)), plt.setp(ax[:,0],ylabel='Z')
        plt.tight_layout()
        plt.savefig('../output/'+params.model_path+'/prediction/data/u/predict_u_'+str(i).zfill(4)+'.png')
        plt.close()
        
        # plot V
        c = 4
        fig, ax = plt.subplots(2,2, figsize=(14,14), sharey=True, sharex=True)
        u1 = ax[0,0].scatter(truth[ID_diag1,1], truth[ID_diag1,2], c=truth[ID_diag1,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        u1p = ax[1,0].scatter(pred[ID_diag1,1], pred[ID_diag1,2], c=pred[ID_diag1,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        u2 = ax[0,1].scatter(truth[ID_diag2,0], truth[ID_diag2,2], c=truth[ID_diag2,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        u2p = ax[1,1].scatter(pred[ID_diag2,0], pred[ID_diag2,2], c=pred[ID_diag2,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        ax[0,0].set_title(r'$v$'), ax[0,1].set_title(r'$v$'), ax[1,0].set_title(r'$\tilde{v}$'), ax[1,1].set_title(r'$\tilde{v}$')
        plt.colorbar(u1), plt.colorbar(u1p), plt.colorbar(u2), plt.colorbar(u2p)                
        plt.setp(ax[-1,0],xlabel=r'$X : |x-y|<$'+str(params.d)), plt.setp(ax[-1,1],xlabel=r'$X : |y-(1-x)|<$'+str(params.d)), plt.setp(ax[:,0],ylabel='Z')
        plt.tight_layout()
        plt.savefig('../output/'+params.model_path+'/prediction/data/v/predict_v_'+str(i).zfill(4)+'.png')
        plt.close()
        
        # plot W
        c = 5
        fig, ax = plt.subplots(2,2, figsize=(14,14), sharey=True, sharex=True)
        u1 = ax[0,0].scatter(truth[ID_diag1,1], truth[ID_diag1,2], c=truth[ID_diag1,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        u1p = ax[1,0].scatter(pred[ID_diag1,1], pred[ID_diag1,2], c=pred[ID_diag1,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        u2 = ax[0,1].scatter(truth[ID_diag2,0], truth[ID_diag2,2], c=truth[ID_diag2,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        u2p = ax[1,1].scatter(pred[ID_diag2,0], pred[ID_diag2,2], c=pred[ID_diag2,c], cmap='seismic', norm=TwoSlopeNorm(vmin=np.min(truth[:, c]), vcenter=0, vmax=np.max(truth[:, c])),s=params.scatter_size)
        ax[0,0].set_title(r'$w$'), ax[0,1].set_title(r'$w$'), ax[1,0].set_title(r'$\tilde{w}$'), ax[1,1].set_title(r'$\tilde{w}$')
        plt.colorbar(u1), plt.colorbar(u1p), plt.colorbar(u2), plt.colorbar(u2p)                
        plt.setp(ax[-1,0],xlabel=r'$X : |x-y|<$'+str(params.d)), plt.setp(ax[-1,1],xlabel=r'$X : |y-(1-x)|<$'+str(params.d)), plt.setp(ax[:,0],ylabel='Z')
        plt.tight_layout()
        plt.savefig('../output/'+params.model_path+'/prediction/data/w/predict_w_'+str(i).zfill(4)+'.png')
        plt.close()
        
        # plot T
        c = 6
        fig, ax = plt.subplots(2,2, figsize=(14,14), sharey=True, sharex=True)
        u1p = ax[0,0].scatter(truth[ID_diag1,1], truth[ID_diag1,2], c=truth[ID_diag1,c], cmap='seismic', norm=TwoSlopeNorm(0),s=params.scatter_size)
        u2p = ax[0,1].scatter(truth[ID_diag2,0], truth[ID_diag2,2], c=truth[ID_diag2,c], cmap='seismic', norm=TwoSlopeNorm(0),s=params.scatter_size)
        u2t = ax[1,0].scatter(pred[ID_diag1,1], pred[ID_diag1,2], c=pred[ID_diag1,c], cmap='seismic', norm=TwoSlopeNorm(0),s=params.scatter_size)
        u2pt = ax[1,1].scatter(pred[ID_diag2,0], pred[ID_diag2,2], c=pred[ID_diag2,c], cmap='seismic', norm=TwoSlopeNorm(0),s=params.scatter_size)
        #ax[0,0].set_title(r'$T$'), ax[0,1].set_title(r'$T$'), 
        ax[1,0].set_title(r'$\tilde{T}$'), ax[1,1].set_title(r'$\tilde{T}$'), ax[0,0].set_title(r'$T$'), ax[0,1].set_title(r'$T$')
        #plt.colorbar(u1), plt.colorbar(u1p), 
        plt.colorbar(u1p), plt.colorbar(u2p), plt.colorbar(u2t), plt.colorbar(u2pt)                
        plt.setp(ax[-1,0],xlabel=r'$X : |x-y|<$'+str(params.d)), plt.setp(ax[-1,1],xlabel=r'$X : |y-(1-x)|<$'+str(params.d)), plt.setp(ax[:,0],ylabel='Z')
        plt.tight_layout()
        plt.savefig('../output/'+params.model_path+'/prediction/data/T/predict_T_'+str(i).zfill(4)+'.png')
        plt.close()
        
        # plot p
        c = 7
        fig, ax = plt.subplots(2,2, figsize=(14,14), sharey=True, sharex=True)
        u1p = ax[0,0].scatter(truth[ID_diag1,1], truth[ID_diag1,2], c=truth[ID_diag1,c], cmap='seismic', norm=TwoSlopeNorm(np.mean(truth[ID_diag1,c])),s=params.scatter_size)
        u2p = ax[0,1].scatter(truth[ID_diag2,0], truth[ID_diag2,2], c=truth[ID_diag2,c], cmap='seismic', norm=TwoSlopeNorm(np.mean(truth[ID_diag2,c])),s=params.scatter_size)
        u2t = ax[1,0].scatter(pred[ID_diag1,1], pred[ID_diag1,2], c=pred[ID_diag1,c], cmap='seismic', norm=TwoSlopeNorm(np.mean(pred[ID_diag1,c])),s=params.scatter_size)
        u2pt = ax[1,1].scatter(pred[ID_diag2,0], pred[ID_diag2,2], c=pred[ID_diag2,c], cmap='seismic', norm=TwoSlopeNorm(np.mean(pred[ID_diag2,c])),s=params.scatter_size)
        #ax[0,0].set_title(r'$T$'), ax[0,1].set_title(r'$T$'), 
        ax[1,0].set_title(r'$\tilde{p}$'), ax[1,1].set_title(r'$\tilde{p}$'), ax[0,0].set_title(r'$p$'), ax[0,1].set_title(r'$p$')
        #plt.colorbar(u1), plt.colorbar(u1p), 
        plt.colorbar(u1p), plt.colorbar(u2p), plt.colorbar(u2t), plt.colorbar(u2pt)                
        plt.setp(ax[-1,0],xlabel=r'$X : |x-y|<$'+str(params.d)), plt.setp(ax[-1,1],xlabel=r'$X : |y-(1-x)|<$'+str(params.d)), plt.setp(ax[:,0],ylabel='Z')
        plt.tight_layout()
        plt.savefig('../output/'+params.model_path+'/prediction/data/p/predict_p_'+str(i).zfill(4)+'.png')
        plt.close()
if __name__ == "__main__":
    main()