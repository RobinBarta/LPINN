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

from PINN_vor import *

# %%

class PredictionParameter():
    # physical parameter
    Ra, Pr = 1e10, 6.9
    # number where the a plane is cut to plot the intersection
    t_steps = 2
    Nx = 64
    # path to data
    data_path = '../data/RBC_dns_1E10_69_t_42_f8.txt'
    # path the the model weights
    model_path, config_name, weight = '1e10_vor', 'config_3', 900

def main(): 
    # define network with config file
    params = PredictionParameter()
    
    # -----------------------------------
    # load data
    # -----------------------------------
    if params.data_path[-3:] == 'txt':
        data = np.loadtxt(params.data_path,skiprows=1)
        data_input, data_output = np.asarray(data[:,0:4],dtype=np.float32), np.asarray(data[:,4:8],dtype=np.float32)
    if params.data_path[-3:] == 'npz':
        data_input, data_output = np.asarray(np.load(params.data_path)['inputs'],dtype=np.float32), np.asarray(np.load(params.data_path)['y_true'],dtype=np.float32)
    
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
    os.makedirs('../output/'+params.model_path+'/prediction/fields', exist_ok = True)
    
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
    # plot mae
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
    # plot cor
    plt.figure(figsize=(15,10))
    plt.semilogy(np.abs(log[:,13]), label='cor_u',c='black')
    plt.semilogy(np.abs(log[:,14]), label='cor_v',c='red')
    plt.semilogy(np.abs(log[:,15]), label='cor_w',c='green')
    plt.semilogy(np.abs(log[:,16]), label='cor_T',c='blue')
    plt.xlabel('epoch')
    plt.ylabel('correlation')
    plt.legend()
    plt.savefig('../output/'+params.model_path+'/prediction/train_cor.png')
    plt.close()
    
    # plot prediction
    for i, t in enumerate(tqdm(np.unique(data_input[:,0])[:params.t_steps],desc='Prediction: ',position=0,leave=True)):
        # prediction at slice
        yi, zi = np.meshgrid(np.linspace(0,1,params.Nx),np.linspace(0,1,params.Nx))
        yi, zi = np.ravel(yi), np.ravel(zi)
        #xi = 0.5*np.ones_like(yi)
        xi = 1-yi.copy()
        ti = t*np.ones_like(yi)
        input_pred = np.vstack([ti,xi,yi,zi]).T
        prediction = pinn(input_pred)
        pred = prediction.numpy() 
        ID = np.argwhere(data_input[:,0]==t)[:,0]
        truth = np.append(data_input[ID,1:],data_output[ID,:],axis=1)
        ID = np.argwhere(np.abs(truth[:,1]-(1-truth[:,0]))<0.05)[:,0]
        #ID = np.argwhere(np.abs(truth[:,0]-truth[:,1])<0.05)[:,0]
               
        # plot UVWT
        fig, ax = plt.subplots(2,5, figsize=(31,13), sharey=True, sharex=True)
        u1 = ax[0,0].scatter(truth[ID,1],truth[ID,2], c=truth[ID,3], cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5),s=5)
        v1 = ax[0,1].scatter(truth[ID,1],truth[ID,2], c=truth[ID,4], cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5),s=5)
        w1 = ax[0,2].scatter(truth[ID,1],truth[ID,2], c=truth[ID,5], cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5),s=5)
        T1 = ax[0,3].scatter(truth[ID,1],truth[ID,2], c=truth[ID,6], cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.2,vmax=0.2),s=5)
        u2 = ax[1,0].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), pred[:,0].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        v2 = ax[1,1].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), pred[:,1].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        w2 = ax[1,2].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), pred[:,2].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        T2 = ax[1,3].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), pred[:,3].reshape(params.Nx,params.Nx), levels=np.linspace(-0.2,0.2,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.2,vmax=0.2))
        ax[0,0].set_title(r'$u$'), ax[0,1].set_title(r'$v$'), ax[0,2].set_title(r'$w$'), ax[0,3].set_title(r'$T$'), ax[0,4].set_title(r'$p$')
        ax[1,0].set_title(r'$\tilde{u}$'), ax[1,1].set_title(r'$\tilde{v}$'), ax[1,2].set_title(r'$\tilde{w}$'), ax[1,3].set_title(r'$\tilde{T}$'), ax[1,4].set_title(r'$\tilde{p}$')
        plt.colorbar(u1), plt.colorbar(v1), plt.colorbar(w1)  , plt.colorbar(T1)#  , plt.colorbar(p1)               
        plt.colorbar(u2), plt.colorbar(v2), plt.colorbar(w2)  , plt.colorbar(T2)#  , plt.colorbar(p2)               
        plt.setp(ax[-1,:],xlabel=r'$Y$'), plt.setp(ax[:,0],ylabel=r'$Z$')
        plt.tight_layout()
        plt.savefig('../output/'+params.model_path+'/prediction/fields/uvwTp_'+str(i).zfill(4)+'.jpg')
        plt.close()
if __name__ == "__main__":
    main()