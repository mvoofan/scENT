import numpy as np
from keras.datasets import mnist as mnist
import os
from numpy.lib import index_tricks
import pandas as pd
import re
import dill as pickle
import math

def my_csv_read(file_name,skip_line):
    ret=[]
    ret=np.array(ret)
    ftem= open(file_name)
    for i in range(skip_line):
        my_str=ftem.readline()
    while True:
        my_str=ftem.readline()
        if not my_str:
            break
        my_str=str(my_str)
        my_str=[int(int_str) for int_str in my_str.split(',')]
        my_str=np.array(my_str)
        my_str=my_str[np.newaxis,...]
        if 0==ret.size:
            ret=my_str
        else:
            ret=np.concatenate( (ret,my_str), axis=0 )
    return ret
    




def get_data(params, data=None):

    ret = {}

    pkl_file=params['pkl_file']

    ftem=open(pkl_file,'rb')
    pkl_data=pickle.load(ftem)
    ftem.close()

    x_train=pkl_data
    #y_train=np.zeros(np.shape(x_train)[0])

    norm_max=x_train.max()
    norm_min=x_train.min()

    #x_train=x_train/norm_max
    #x_train=(x_train-norm_min)/(norm_max-norm_min)

    x_train_tmp=np.zeros_like(x_train)
    x_train_tmp[x_train<0]=norm_max+np.abs(x_train[x_train<0])
    x_train_tmp[x_train>0]=x_train[x_train>0]
    x_train=x_train_tmp/(norm_max+abs(norm_min))


    #select top  3000
    std_np=np.zeros( np.shape(x_train)[0] )
    for i in range( np.shape(x_train)[0] ):
        std_np[i]= np.std(x_train[i,:])
    

    std_np_t=std_np.copy()
    std_np_t_sort=np.sort(std_np_t)

    x_train = x_train[np.where(std_np>std_np_t_sort[-3001])[0],:]

    y_train=np.zeros(np.shape(x_train)[0])
    




    ret['spectral'] = {}

    #y_train=y_train-1

    


    x_train=x_train[...,np.newaxis]

    ret['spectral']['train_and_test'] = (x_train, y_train, x_train, y_train, x_train, y_train)

    return ret








    '''
    x_train=x_train[...,np.newaxis] #(sampels, 120, 1)

    ret['spectral']['train_and_test'] = (x_train[0:-1000,:,:], y_train[0:-1000], x_train[-1000:,:,:], y_train[-1000:], x_train[-1000:,:,:], y_train[-1000:])


    return ret
    '''
