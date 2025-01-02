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
    Ra, Pr = 1e9, 6.9
    # number where the a plane is cut to plot the intersection
    t_steps = 28
    Nx = 64
    # path to data
    data_path = '../data/RBC_exp_1E9_69_t_27.txt'
    # path the the model weights
    model_path, config_name, weight = '1e9_exp_new4', 'config_2', 20000

def main(): 
    # define network with config file
    params = PredictionParameter()
    
    # -----------------------------------
    # load data
    # -----------------------------------
    if params.data_path[-3:] == 'txt':
        data = np.loadtxt(params.data_path,skiprows=1)
        data_input, data_output = np.asarray(data[:,0:4],dtype=np.float32), np.asarray(data[:,4:9],dtype=np.float32)
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
    plt.semilogy(log[:,13], label='mae_p',c='purple')
    plt.xlabel('epoch')
    plt.ylabel('mae')
    plt.legend()
    plt.savefig('../output/'+params.model_path+'/prediction/train_mae.png')
    plt.close()
    # plot cor
    plt.figure(figsize=(15,10))
    plt.semilogy(np.abs(log[:,14]), label='cor_u',c='black')
    plt.semilogy(np.abs(log[:,15]), label='cor_v',c='red')
    plt.semilogy(np.abs(log[:,16]), label='cor_w',c='green')
    plt.semilogy(np.abs(log[:,17]), label='cor_T',c='blue')
    plt.semilogy(np.abs(log[:,18]), label='cor_p',c='purple')
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
        truth[:,-1] = np.asarray(np.loadtxt('../data/RBC_dns_1E10_69_t_100_f8.txt',skiprows=1)[ID,-1],dtype=np.float32)
        ID = np.argwhere(np.abs(truth[:,1]-(1-truth[:,0]))<0.05)[:,0]
        #ID = np.argwhere(np.abs(truth[:,0]-truth[:,1])<0.05)[:,0]
               
        # plot UVWTP
        fig, ax = plt.subplots(2,5, figsize=(31,13), sharey=True, sharex=True)
        u1 = ax[0,0].scatter(truth[ID,1],truth[ID,2], c=truth[ID,3], cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5),s=5)
        v1 = ax[0,1].scatter(truth[ID,1],truth[ID,2], c=truth[ID,4], cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5),s=5)
        w1 = ax[0,2].scatter(truth[ID,1],truth[ID,2], c=truth[ID,5], cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5),s=5)
        T1 = ax[0,3].scatter(truth[ID,1],truth[ID,2], c=truth[ID,6], cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.2,vmax=0.2),s=5)
        p1 = ax[0,4].scatter(truth[ID,1],truth[ID,2], c=truth[ID,7], cmap='seismic', norm=TwoSlopeNorm(np.mean(truth[ID,7]),vmin=np.min(truth[ID,7]),vmax=np.max(truth[ID,7])),s=5)
        u2 = ax[1,0].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), pred[:,0].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        v2 = ax[1,1].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), pred[:,1].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        w2 = ax[1,2].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), pred[:,2].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        T2 = ax[1,3].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), pred[:,3].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        p2 = ax[1,4].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), pred[:,4].reshape(params.Nx,params.Nx), levels=np.linspace(np.min(pred[:,4]),np.max(pred[:,4]),801), cmap='seismic', norm=TwoSlopeNorm(np.median(pred[:,4]),vmin=np.min(pred[:,4]),vmax=np.max(pred[:,4])))
        ax[0,0].set_title(r'$u$'), ax[0,1].set_title(r'$v$'), ax[0,2].set_title(r'$w$'), ax[0,3].set_title(r'$T$'), ax[0,4].set_title(r'$p$')
        ax[1,0].set_title(r'$\tilde{u}$'), ax[1,1].set_title(r'$\tilde{v}$'), ax[1,2].set_title(r'$\tilde{w}$'), ax[1,3].set_title(r'$\tilde{T}$'), ax[1,4].set_title(r'$\tilde{p}$')
        plt.colorbar(u1), plt.colorbar(v1), plt.colorbar(w1)  , plt.colorbar(T1)  , plt.colorbar(p1)               
        plt.colorbar(u2), plt.colorbar(v2), plt.colorbar(w2)  , plt.colorbar(T2)  , plt.colorbar(p2)               
        plt.setp(ax[-1,:],xlabel=r'$Y$'), plt.setp(ax[:,0],ylabel=r'$Z$')
        plt.tight_layout()
        plt.savefig('../output/'+params.model_path+'/prediction/fields/uvwTp_'+str(i).zfill(4)+'.jpg')
        plt.close()
        
        # plot derivatives
        Wt, Wx, Wy, Wz, Pz, Wxx, Wyy, Wzz, U, V, W, Ti = np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        Tt, Tx, Ty, Tz = np.empty(0), np.empty(0), np.empty(0), np.empty(0)
        Txx, Tyy, Tzz = np.empty(0), np.empty(0), np.empty(0)
        for ij in range(int(len(input_pred)/20000)+1):
            x = tf.Variable(input_pred[ij*20000:(ij+1)*20000,:])
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
                T_xx = tape.gradient(w_x, x)[...,1]
                T_yy = tape.gradient(w_y, x)[...,2]
                T_zz = tape.gradient(w_z, x)[...,3]    
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
            Txx = np.append(Txx,T_xx.numpy())
            Tyy = np.append(Tyy,T_yy.numpy())
            Tzz = np.append(Tzz,T_zz.numpy())
            Wxx = np.append(Wxx,w_xx.numpy())
            Wyy = np.append(Wyy,w_yy.numpy())
            Wzz = np.append(Wzz,w_zz.numpy())
            Pz = np.append(Pz,p_z.numpy())
        T_pred = Wt + U*Wx + V*Wy + W*Wz + Pz - np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)
        
        fig, ax = plt.subplots(2,5, figsize=(31,13), sharey=True, sharex=True)
        u1 = ax[0,0].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), Wt.reshape(params.Nx,params.Nx), levels=np.linspace(np.min(Wt),np.max(Wt),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Wt),vmax=np.max(Wt)))
        v1 = ax[0,1].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), (U*Wx + V*Wy + W*Wz).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((U*Wx + V*Wy + W*Wz)),np.max((U*Wx + V*Wy + W*Wz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((U*Wx + V*Wy + W*Wz)),vmax=np.max((U*Wx + V*Wy + W*Wz))))
        w1 = ax[0,2].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), -(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)).reshape(params.Nx,params.Nx), levels=np.linspace(np.min(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))),np.max(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))),vmax=np.max(-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))))
        T1 = ax[0,3].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), Pz.reshape(params.Nx,params.Nx), levels=np.linspace(np.min(Pz),np.max(Pz),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Pz),vmax=np.max(Pz)))
        p1 = ax[0,4].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), T_pred.reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        u2 = ax[1,0].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), (Wt+(U*Wx + V*Wy + W*Wz)).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Wt+(U*Wx + V*Wy + W*Wz))),np.max((Wt+(U*Wx + V*Wy + W*Wz))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Wt+(U*Wx + V*Wy + W*Wz))),vmax=np.max((Wt+(U*Wx + V*Wy + W*Wz)))))
        v2 = ax[1,1].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), 0*(Wt+(U*Wx + V*Wy + W*Wz)+Pz).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),vmax=np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz))))
        w2 = ax[1,2].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), (Wt+(U*Wx + V*Wy + W*Wz)+Pz).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),vmax=np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz))))
        T2 = ax[1,3].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), (Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))),np.max((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))),vmax=np.max((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))))))
        p2 = ax[1,4].scatter(truth[ID,1],truth[ID,2], c=truth[ID,6], cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5),s=5)
        ax[0,0].set_title(r'$\partial_t w$'), ax[0,1].set_title(r'$\vec{u}\cdot\nabla w$'), ax[0,2].set_title(r'$-(Pr/Ra)^{0.5} \Delta w$'), ax[0,3].set_title(r'$\partial_z p$'), ax[0,4].set_title(r'$\Sigma$')
        ax[1,0].set_title(r'$D_t w$'), ax[1,1].set_title('None'), ax[1,2].set_title(r'$D_t w + \partial_z p$'), ax[1,3].set_title(r'$D_t w - (Pr/Ra)^{0.5} \Delta w$'), ax[1,4].set_title(r'$T$')
        plt.colorbar(u1), plt.colorbar(v1), plt.colorbar(w1)  , plt.colorbar(T1)  , plt.colorbar(p1)               
        plt.colorbar(u2), plt.colorbar(v2), plt.colorbar(w2)  , plt.colorbar(T2)  , plt.colorbar(p2)               
        plt.setp(ax[-1,:],xlabel=r'$Y$'), plt.setp(ax[:,0],ylabel=r'$Z$')
        plt.tight_layout()
        plt.savefig('../output/'+params.model_path+'/prediction/fields/NSE_'+str(i).zfill(4)+'.jpg')
        plt.close()
        
        fig, ax = plt.subplots(2,5, figsize=(31,13), sharey=True, sharex=True)
        u1 = ax[0,0].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), Tt.reshape(params.Nx,params.Nx), levels=np.linspace(np.min(Tt),np.max(Tt),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(Tt),vmax=np.max(Tt)))
        v1 = ax[0,1].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), (U*Tx + V*Ty + W*Tz).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((U*Tx + V*Ty + W*Tz)),np.max((U*Tx + V*Ty + W*Tz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((U*Tx + V*Ty + W*Tz)),vmax=np.max((U*Tx + V*Ty + W*Tz))))
        w1 = ax[0,2].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), -(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz)).reshape(params.Nx,params.Nx), levels=np.linspace(np.min(-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz))),np.max(-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min(-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz))),vmax=np.max(-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz)))))
        T1 = ax[0,3].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), (Tt+(U*Tx + V*Ty + W*Tz)-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz))).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Tt+(U*Tx + V*Ty + W*Tz)-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz)))),np.max((Tt+(U*Tx + V*Ty + W*Tz)-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz)))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Tt+(U*Tx + V*Ty + W*Tz)-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz)))),vmax=np.max((Tt+(U*Tx + V*Ty + W*Tz)-(np.sqrt(1/(params.Pr*params.Ra)) * (Txx+Tyy+Tzz))))))
        p1 = ax[0,4].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), pred[:,3].reshape(params.Nx,params.Nx), levels=np.linspace(-0.5,0.5,801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5))
        u2 = ax[1,0].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), (Tt+(U*Tx + V*Ty + W*Tz)).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Tt+(U*Tx + V*Ty + W*Tz))),np.max((Tt+(U*Tx + V*Ty + W*Tz))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Tt+(U*Tx + V*Ty + W*Tz))),vmax=np.max((Tt+(U*Tx + V*Ty + W*Tz)))))
        v2 = ax[1,1].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), 0*(Wt+(U*Wx + V*Wy + W*Wz)+Pz).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),vmax=np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz))))
        w2 = ax[1,2].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), 0*(Wt+(U*Wx + V*Wy + W*Wz)+Pz).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Wt+(U*Wx + V*Wy + W*Wz)+Pz)),vmax=np.max((Wt+(U*Wx + V*Wy + W*Wz)+Pz))))
        T2 = ax[1,3].contourf(yi.reshape(params.Nx,params.Nx),zi.reshape(params.Nx,params.Nx), 0*(Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))).reshape(params.Nx,params.Nx), levels=np.linspace(np.min((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))),np.max((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))),801), cmap='seismic', norm=TwoSlopeNorm(0,vmin=np.min((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz)))),vmax=np.max((Wt+(U*Wx + V*Wy + W*Wz)-(np.sqrt(params.Pr/params.Ra) * (Wxx+Wyy+Wzz))))))
        p2 = ax[1,4].scatter(truth[ID,1],truth[ID,2], c=truth[ID,6], cmap='seismic', norm=TwoSlopeNorm(0,vmin=-0.5,vmax=0.5),s=5)
        ax[0,0].set_title(r'$\partial_t T$'), ax[0,1].set_title(r'$\vec{u}\cdot\nabla T$'), ax[0,2].set_title(r'$-(Pr Ra)^{-0.5} \Delta T$'), ax[0,3].set_title(r'$\Sigma$'), ax[0,4].set_title(r'$T$')
        ax[1,0].set_title(r'$D_t T$'), ax[1,1].set_title('None'), ax[1,2].set_title('None'), ax[1,3].set_title('None'), ax[1,4].set_title(r'$T$')
        plt.colorbar(u1), plt.colorbar(v1), plt.colorbar(w1)  , plt.colorbar(T1)  , plt.colorbar(p1)               
        plt.colorbar(u2), plt.colorbar(v2), plt.colorbar(w2)  , plt.colorbar(T2)  , plt.colorbar(p2)               
        plt.setp(ax[-1,:],xlabel=r'$Y$'), plt.setp(ax[:,0],ylabel=r'$Z$')
        plt.tight_layout()
        plt.savefig('../output/'+params.model_path+'/prediction/fields/EE_'+str(i).zfill(4)+'.jpg')
        plt.close()
if __name__ == "__main__":
    main()