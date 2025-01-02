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
    Ra, Pr = 1e8, 0.7
    # number where the a plane is cut to plot the intersection
    t_step = 0
    Nt = 9
    Nx = 386
    N = 150
    # path to data
    data_path = '../data/RBC_dns_1e8_07_t_9.npz'
    # path the the model weights
    model_path, config_name, weight0, weight1 = '1e8_best', 'config_2', 1, 60

def main(): 
    # define network with config file
    params = PredictionParameter()
    
    # -----------------------------------
    # load data
    # -----------------------------------
    data_input, data_output = np.asarray(np.load(params.data_path)['inputs'],dtype=np.float32), np.asarray(np.load(params.data_path)['y_true'],dtype=np.float32)
    # preprocessing
    data = DataProcessor([data_input, data_output]).process()
    data_input, data_output = data[0], data[1]
    
    # load config
    spec = importlib.util.spec_from_file_location("config", '../output/'+params.model_path+'/'+params.config_name+'.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    params2 = config.Parameter()
    
    # define output folder
    os.makedirs('../output/'+params.model_path+'/prediction', exist_ok = True)
    os.makedirs('../output/'+params.model_path+'/prediction/epochs', exist_ok = True)
    
    # -------------------
    # build PINN model
    # -------------------   
    pinn = PINN(params2)
    # create a dummy input to build the model, this call creates the model's variables
    _ = pinn(tf.zeros((1, 4)))  
    
    t = np.unique(data_input[:,0])[params.t_step]
    for weight in tqdm(np.linspace(params.weight0,params.weight1,params.weight1-params.weight0+1)):
        pinn.load_weights('../output/'+params.model_path+'/weights/weights_epoch_'+f"{int(weight):04d}"+'.weights.h5')
        
        # prediction at slice
        ti = data_input[:,0].reshape(params.Nx,params.Nx,params.Nx,params.Nt)[params.N,:,:,params.t_step].ravel()
        xi = data_input[:,1].reshape(params.Nx,params.Nx,params.Nx,params.Nt)[params.N,:,:,params.t_step].ravel()
        yi = data_input[:,2].reshape(params.Nx,params.Nx,params.Nx,params.Nt)[params.N,:,:,params.t_step].ravel()
        zi = data_input[:,3].reshape(params.Nx,params.Nx,params.Nx,params.Nt)[params.N,:,:,params.t_step].ravel()
        input_pred = np.vstack([ti,xi,yi,zi]).T
        prediction = pinn(input_pred)
        
        # filter by time step
        ID = np.argwhere(data_input[:,0]==t)[:,0]
        truth = np.append(data_input[ID,1:],data_output[ID,:],axis=1)
        ID = np.argwhere(input_pred[:,0]==t)[:,0]
        pred = prediction.numpy()[ID,:]
        
        # plot UVWTP
        fig, ax = plt.subplots(2,5, figsize=(31,13), sharey=True, sharex=True)
        u1 = ax[0,0].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], truth[:,3].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        v1 = ax[0,1].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], truth[:,4].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        w1 = ax[0,2].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], truth[:,5].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        T1 = ax[0,3].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], truth[:,6].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        p1 = ax[0,4].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], truth[:,7].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], levels=np.linspace(np.min(truth[:,7]),np.max(truth[:,7]),801), cmap='seismic', norm=TwoSlopeNorm(np.mean(truth[:,7]),vmin=np.min(truth[:,7]),vmax=np.max(truth[:,7])))
        u2 = ax[1,0].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], pred[:,0].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        v2 = ax[1,1].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], pred[:,1].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        w2 = ax[1,2].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], pred[:,2].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        T2 = ax[1,3].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], pred[:,3].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        p2 = ax[1,4].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], pred[:,4].reshape(params.Nx,params.Nx), levels=np.linspace(np.min(truth[:,7]),np.max(truth[:,7]),801), cmap='seismic', norm=TwoSlopeNorm(np.mean(truth[:,7]),vmin=np.min(truth[:,7]),vmax=np.max(truth[:,7])))
        ax[0,0].set_title(r'$u$'), ax[0,1].set_title(r'$v$'), ax[0,2].set_title(r'$w$'), ax[0,3].set_title(r'$T$'), ax[0,4].set_title(r'$p$')
        ax[1,0].set_title(r'$\tilde{u}$'), ax[1,1].set_title(r'$\tilde{v}$'), ax[1,2].set_title(r'$\tilde{w}$'), ax[1,3].set_title(r'$\tilde{T}$'), ax[1,4].set_title(r'$\tilde{p}$')
        plt.colorbar(u1), plt.colorbar(v1), plt.colorbar(w1)  , plt.colorbar(T1)  , plt.colorbar(p1)               
        plt.colorbar(u2), plt.colorbar(v2), plt.colorbar(w2)  , plt.colorbar(T2)  , plt.colorbar(p2)               
        plt.setp(ax[-1,:],xlabel=r'$Y$'), plt.setp(ax[:,0],ylabel=r'$Z$')
        plt.tight_layout()
        plt.savefig('../output/'+params.model_path+'/prediction/epochs/uvwTp_'+str(weight).zfill(4)+'.jpg')
        plt.close()
        
        # plot derivatives
        Wt, Wx, Wy, Wz, Pz, Wxx, Wyy, Wzz, U, V, W, Ti = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        Tt, Tx, Ty, Tz = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        Txx, Tyy, Tzz = np.empty(0), np.empty(0), np.empty(0)
        for ij in range(int(len(input_pred[ID,:])/20000)+1):
            x = tf.Variable(input_pred[ID,:][ij*20000:(ij+1)*20000,:])
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x)
                data_watch = pinn(x)
                u, v, w, T, p = tf.unstack(data_watch, axis=1)
                # first derivatives
                w_t, w_x, w_y, w_z = tf.unstack(tape.gradient(w, x), axis=-1)
                T_t, T_x, T_y, T_z = tf.unstack(tape.gradient(T, x), axis=-1)
                p_z = tape.gradient(p, x)[...,3]
                # second derivatives
                w_xx = tape.gradient(w_x, x)[...,1]
                w_yy = tape.gradient(w_y, x)[...,2]
                w_zz = tape.gradient(w_z, x)[...,3]  
   
                del tape
            U = np.append(U,u.numpy())
            V = np.append(V,v.numpy())
            W = np.append(W,w.numpy())
            Ti = np.append(Ti,T.numpy())
            Wt = np.append(Wt,w_t.numpy())
            Wx = np.append(Wx,w_x.numpy())
            Wy = np.append(Wy,w_y.numpy())
            Wz = np.append(Wz,w_z.numpy())
            Tt = np.append(Tt,T_t.numpy())
            Tx = np.append(Tx,T_x.numpy())
            Ty = np.append(Ty,T_y.numpy())
            Tz = np.append(Tz,T_z.numpy())
            Wxx = np.append(Wxx,w_xx.numpy())
            Wyy = np.append(Wyy,w_yy.numpy())
            Wzz = np.append(Wzz,w_zz.numpy())
            Pz = np.append(Pz,p_z.numpy())
        T_pred = Wt + U*Wx + V*Wy + W*Wz + Pz - np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)
        
        fig, ax = plt.subplots(2,5, figsize=(31,13), sharey=True, sharex=True)
        u1 = ax[0,0].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], Wt.reshape(params.Nx,params.Nx), levels=np.linspace(np.min(Wt),np.max(Wt),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Wt),vmax=np.max(Wt)))
        v1 = ax[0,1].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], (U*Wx + V*Wy + W*Wz).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((U*Wx + V*Wy + W*Wz)),np.max((U*Wx + V*Wy + W*Wz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((U*Wx + V*Wy + W*Wz)),vmax=np.max((U*Wx + V*Wy + W*Wz))))
        w1 = ax[0,2].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], -(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)).reshape(params.Nx,params.Nx), levels=np.linspace(np.min(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))),np.max(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))),vmax=np.max(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))))
        T1 = ax[0,3].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], Pz.reshape(params.Nx,params.Nx), levels=np.linspace(np.min(Pz),np.max(Pz),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Pz),vmax=np.max(Pz)))
        p1 = ax[0,4].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], T_pred.reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        u2 = ax[1,0].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], (Wt+(U*Wx + V*Wy + W*Wz)).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Wt+(U*Wx + V*Wy + W*Wz))),np.max((Wt+(U*Wx + V*Wy + W*Wz))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Wt+(U*Wx + V*Wy + W*Wz))),vmax=np.max((Wt+(U*Wx + V*Wy + W*Wz)))))
        v2 = ax[1,1].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], 0*(Wt+(U*Wx + V*Wy + W*Wz)+Pz).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),vmax=np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz))))
        w2 = ax[1,2].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], (Wt+(U*Wx + V*Wy + W*Wz)+Pz).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),vmax=np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz))))
        T2 = ax[1,3].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], (Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))),np.max((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))),vmax=np.max((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))))))
        p2 = ax[1,4].contourf(truth[:,1].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:],truth[:,2].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], truth[:,6].reshape(params.Nx,params.Nx,params.Nx)[params.N,:,:], levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        ax[0,0].set_title(r'$\partial_t w$'), ax[0,1].set_title(r'$\vec{u}\cdot\nabla w$'), ax[0,2].set_title(r'$-(Pr/Ra)^{0.5} \Delta w$'), ax[0,3].set_title(r'$\partial_z p$'), ax[0,4].set_title(r'$\Sigma$')
        ax[1,0].set_title(r'$D_t w$'), ax[1,1].set_title('None'), ax[1,2].set_title(r'$D_t w + \partial_z p$'), ax[1,3].set_title(r'$D_t w - (Pr/Ra)^{0.5} \Delta w$'), ax[1,4].set_title(r'$T$')
        plt.colorbar(u1), plt.colorbar(v1), plt.colorbar(w1)  , plt.colorbar(T1)  , plt.colorbar(p1)               
        plt.colorbar(u2), plt.colorbar(v2), plt.colorbar(w2)  , plt.colorbar(T2)  , plt.colorbar(p2)               
        plt.setp(ax[-1,:],xlabel=r'$Y$'), plt.setp(ax[:,0],ylabel=r'$Z$')
        plt.tight_layout()
        plt.savefig('../output/'+params.model_path+'/prediction/epochs/NSE_'+str(weight).zfill(4)+'.jpg')
        plt.close()
if __name__ == "__main__":
    main()