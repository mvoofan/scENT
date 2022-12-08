
import sys, os
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse
from collections import defaultdict
import tensorflow as tf

from core.ad_mic_data2_top3000 import get_data
from applications.my_ADDC2_unified import run_net, run_net_initial

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--dset', type=str, help='gpu number to use', default='mnist')
args = parser.parse_args()

# SELECT GPU


if not os.path.exists( os.path.join('./applications/ad_mic_data/top3000/', 'cluster6_save_data1'    ) ):
    os.makedirs( os.path.join('./applications/ad_mic_data/top3000/', 'cluster6_save_data1'   ) )


pkl_file=os.path.join('./core/','AD_mic_data.pkl' )
save_dir=os.path.join('./applications/ad_mic_data/top3000/', 'cluster6_save_data1')


params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {
        'dset': args.dset,                 
        }
params.update(general_params)

# SET DATASET SPECIFIC HYPERPARAMETERS
if args.dset == 'mnist':
    mnist_params = {
        'n_clusters': 6,                   # number of clusters in data
        'n_nbrs': 3,                       # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
        'scale_nbr': 2,                     # neighbor used to determine scale of gaussian graph Laplacian; calculated by
        'batch_size': 1000,                 # batch size for spectral net 1000, 20 can run 
        'use_approx': False,                # enable / disable approximate nearest neighbors
        'use_all_data': True,               # enable to use all data for training (no test set)
        'latent_dim': 64,
        'spec_lr': 1e-11,                    #le-3 1e-7
        'img_dim': 28,
        'filters': 16,
        'training_flag': 'defense',          # 'defense' or 'advsior' 
        'feather_len':  384,
        'train_or_test': 'train',
        'iter_num':    1,
        'gpu':    2,
        'pkl_file':    pkl_file,
        'save_dir':    save_dir,
        'data_ind':    1,
        'gpu_ids':    '0'
        }
    params.update(mnist_params)


data = get_data(params)   #载入数据，默认MINST x_train, y_train, x_val, y_val, x_test, y_test

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session =tf.Session(config=config)

#run_net_initial(data, params)


for i in range(10):

    params['iter_num']=i+1

    params['training_flag']='defense'

    # RUN EXPERIMENT
    if 'train'==params['train_or_test']:
        run_net(data, params)
    elif 'test'==params['train_or_test']:
        run_net(data, params)   #TOADD test_net


    
    params['training_flag']='advsior'

    # RUN EXPERIMENT
    if 'train'==params['train_or_test']:
        run_net(data, params)
    elif 'test'==params['train_or_test']:
        run_net(data, params)   #TOADD test_net

