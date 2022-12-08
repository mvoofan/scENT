import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


import numpy as np
from sklearn import manifold
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import matplotlib.pyplot as plt
from keras.layers import Input
from core import my_Conv2_unified
import scipy.io as scio
import tensorflow as tf
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from keras.utils import to_categorical
import imageio
from keras.models import Model
from core.util import get_scale, print_accuracy, get_cluster_sols, LearningHandler, make_layer_list, train_gen, get_y_preds, print_accuracy_to_fo
import cv2
import tensorflow as tf
from keras import backend as K
from sklearn import metrics
from jqmcvi import base
import dill as pickle
from umap import UMAP
#from sklearn.metrics import silhouette_score

def my_dunn_index(x_train, labels):
    new_x_train=np.concatenate( (x_train, labels[...,np.newaxis]), axis=1 )
    #new_x_train= x_train.copy()
    n_clusters=int(labels.max()+1)
    my_dunn_list=[]
    for i in range(n_clusters):
        data_to_add=new_x_train[np.where(i==labels), :]
        data_to_add=np.squeeze(data_to_add)
        if not 0==data_to_add.size:
            my_dunn_list.append(data_to_add)
    return base.dunn(my_dunn_list)

def run_net_initial(data, params):
    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu_ids']


    pkl_file= params['pkl_file']

    ftem=open(pkl_file, 'rb')
    pkl_data=pickle.load(ftem)
    ftem.close()

    x_train_original=pkl_data
    y_train_original=np.zeros(np.shape(pkl_data)[0])



    x_train_unlabeled, y_train_unlabeled, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']

    save_dir=params['save_dir']

    zzz=200  #200

    
    K.clear_session()






    #inputs_vae = Input(shape=(params['img_dim'],params['img_dim'],1), name='inputs_vae')
    inputs_vae = Input(shape=(params['feather_len'],1), name='inputs_vae')
    ConvAE = my_Conv2_unified.ConvAE(inputs_vae,params)






    for j in range(10):

        lh = LearningHandler(lr=params['spec_lr'], drop=params['spec_drop'], lr_tensor=ConvAE.learning_rate,
                                patience=params['spec_patience'])


        lh.on_train_begin()
        # one_hots = to_categorical(y_val,10)
        losses_vae = np.empty((zzz,))
        acc = np.empty((zzz,))
        losse = np.empty((zzz,))
        nmi1 = np.empty((zzz,))
        noise = 1*np.random.rand(np.shape(x_train_unlabeled)[0],params['latent_dim'])
        noise1 = 1 * np.random.rand(np.shape(x_train_unlabeled)[0], params['n_clusters'])


        for i in range(200):






            x_val_t = ConvAE.encoder.predict(x_val)
            # scale = conv1.get_scale(x_val_y, 1000, params['scale_nbr'])
            x_val_t1 = ConvAE.Advsior.predict(x_val)  #加扰动
            # q= target_distribution(x_val_y)
            x_sp = ConvAE.classfier.predict(x_val_t)
            y_sp = x_sp.argmax(axis=1)
            x_val_y = ConvAE.classfier.predict(x_val_t1+x_val_t)
            y_sp_1 = x_val_y.argmax(axis=1)


            x_val_1 = ConvAE.decoder.predict(x_val_t)
            x_val_2 = ConvAE.decoder.predict(x_val_t1+x_val_t)



            accuracy = print_accuracy(y_sp, y_val, params['n_clusters'])

            if 0==(i+1)%10 and 'defense'==params['training_flag']:
                
                ari=metrics.adjusted_rand_score(y_val, y_sp)

                print('ari:    %f'    %    ari)

            
            nmi1[i] = accuracy
            accuracy = print_accuracy(y_sp_1, y_val, params['n_clusters'])

            losses_vae[i] = ConvAE.train_initial(x_train_unlabeled,noise,noise1,params['batch_size'])  # trainning defense            


            
            print("1Z Epoch: {}, loss={:2f},D = removed".format(i, losses_vae[i]))  #remove M
            acc[i] = accuracy

            # nmi1[i] = nmi(y_sp, y_val)
            print('NMI: ' + str(np.round(nmi(y_sp, y_val), 4)))

            print('NMI: ' + str(np.round(nmi(y_sp_1, y_val), 4)))

            if i>1:
                if np.abs(losses_vae[i]-losses_vae[i-1])<0.0001:
                    print('STOPPING EARLY')
                    break


    for j in range(10):


        
        lh = LearningHandler(lr=params['spec_lr'], drop=params['spec_drop'], lr_tensor=ConvAE.learning_rate,
                                patience=params['spec_patience'])


        lh.on_train_begin()
        # one_hots = to_categorical(y_val,10)
        losses_vae = np.empty((zzz,))
        acc = np.empty((zzz,))
        losse = np.empty((zzz,))
        nmi1 = np.empty((zzz,))
        noise = 1*np.random.rand(np.shape(x_train_unlabeled)[0],params['latent_dim'])
        noise1 = 1 * np.random.rand(np.shape(x_train_unlabeled)[0], params['n_clusters'])


        for i in range(200):




            x_val_t = ConvAE.encoder.predict(x_val)
            # scale = conv1.get_scale(x_val_y, 1000, params['scale_nbr'])
            x_val_t1 = ConvAE.Advsior.predict(x_val)  #加扰动
            # q= target_distribution(x_val_y)
            x_sp = ConvAE.classfier.predict(x_val_t)
            y_sp = x_sp.argmax(axis=1)
            x_val_y = ConvAE.classfier.predict(x_val_t1+x_val_t)
            y_sp_1 = x_val_y.argmax(axis=1)


            x_val_1 = ConvAE.decoder.predict(x_val_t)
            x_val_2 = ConvAE.decoder.predict(x_val_t1+x_val_t)



            accuracy = print_accuracy(y_sp, y_val, params['n_clusters'])

            if 0==(i+1)%10 and 'defense'==params['training_flag']:
                
                ari=metrics.adjusted_rand_score(y_val, y_sp)

                print('ari:    %f'    %    ari)

            
            nmi1[i] = accuracy
            accuracy = print_accuracy(y_sp_1, y_val, params['n_clusters'])

            losses_vae[i] = ConvAE.train_initial2(x_train_unlabeled,noise,noise1,params['batch_size'])  # trainning defense            


            
            print("1Z Epoch: {}, loss={:2f},D = removed".format(i, losses_vae[i]))  #remove M
            acc[i] = accuracy

            # nmi1[i] = nmi(y_sp, y_val)
            print('NMI: ' + str(np.round(nmi(y_sp, y_val), 4)))

            print('NMI: ' + str(np.round(nmi(y_sp_1, y_val), 4)))

            if i>1:
                if np.abs(losses_vae[i]-losses_vae[i-1])<0.0001:
                    print('STOPPING EARLY')
                    break
        

        
    x_val_t = ConvAE.encoder.predict(x_val)

    ###

    shuffle_ind=np.random.permutation(  range(np.shape(x_val_t)[0])  )


    x_val_t_permuted=x_val_t[shuffle_ind,:]
    y_sp_permuted=y_sp[shuffle_ind]

    ###
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(x_val_t_permuted)
    fig = plt.figure()
    #plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=y_val, cmap=plt.cm.get_cmap("jet", 10))
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=y_sp_permuted, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.savefig(os.path.join(save_dir, 'iter0_initial_phase.svg' ) )
    plt.savefig(os.path.join(save_dir, 'iter0_initial_phase.png'  ) )

    #add UMAP
    umap_2d=UMAP(n_components=2, init='random', random_state=0 )
    proj_2d=umap_2d.fit_transform(x_val_t_permuted)
    fig2=plt.figure()
    #plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=2, c=y_val, cmap=plt.cm.get_cmap("jet", 10))
    plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=2, c=y_sp_permuted, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.savefig(os.path.join(save_dir, 'iter0_initial_phase_umap.svg'  ) )
    plt.savefig(os.path.join(save_dir, 'iter0_initial_phase_umap.png'  ) )


    print("finished training")



    labels= y_sp
    write_to_pkl_file=os.path.join(save_dir, 'labels_y_pred_data_initial.pkl')
    ftem=open(write_to_pkl_file, 'wb')
    pickle.dump(labels, ftem)
    ftem.close()      
    ConvAE.vae.save_weights(os.path.join(save_dir, 'my_ave_iter0.h5'    ) )
    ConvAE.Discriminator.save_weights(os.path.join(save_dir, 'my_discriminator_iter0.h5'    ))



    x_val_y = ConvAE.vae.predict(x_val)[2]
    y_sp = x_val_y.argmax(axis=1)

    print_accuracy(y_sp, y_val, params['n_clusters'])

    nmi_score1 = nmi(y_sp, y_val)
    print('NMI: ' + str(np.round(nmi_score1, 4)))



def run_net(data, params):

    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu_ids']


    pkl_file= params['pkl_file']

    ftem=open(pkl_file, 'rb')
    pkl_data=pickle.load(ftem)
    ftem.close()

    x_train_original=pkl_data
    y_train_original=np.zeros(np.shape(pkl_data)[0])



    x_train_unlabeled, y_train_unlabeled, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']

    save_dir=params['save_dir']

    zzz=200  #200

    
    K.clear_session()






    #inputs_vae = Input(shape=(params['img_dim'],params['img_dim'],1), name='inputs_vae')
    inputs_vae = Input(shape=(params['feather_len'],1), name='inputs_vae')
    ConvAE = my_Conv2_unified.ConvAE(inputs_vae,params)
    if 'defense'==params['training_flag']:
        if os.path.exists(  os.path.join(save_dir, 'my_ADV_iter%d.h5'    %    (params['iter_num']-1))  ):
            ConvAE.Advsior.load_weights(  os.path.join(save_dir, 'my_ADV_iter%d.h5'    %    (params['iter_num']-1))  )
            print('\033[1;31m ADV iter %d is loaded\033[0m' % (params['iter_num']-1 ))
        if os.path.exists(os.path.join(save_dir, 'my_ave_iter%d.h5'    %    (params['iter_num']-1))):
            ConvAE.vae.load_weights(os.path.join(save_dir, 'my_ave_iter%d.h5'    %    (params['iter_num']-1)))
            print('\033[1;31m ave iter %d is loaded\033[0m' % (params['iter_num']-1) )
        if os.path.exists(os.path.join(save_dir, 'my_discriminator_iter%d.h5'    %    (params['iter_num']-1))):
            ConvAE.Discriminator.load_weights(os.path.join(save_dir, 'my_discriminator_iter%d.h5'    %    (params['iter_num']-1)))
            print('\033[1;31m discriminator iter %d is loaded\033[0m' % (params['iter_num']-1) )
    elif 'advsior'==params['training_flag']:
        if os.path.exists( os.path.join(save_dir, 'my_ADV_iter%d.h5'    %    (params['iter_num']-1)  )):
            ConvAE.Advsior.load_weights(  os.path.join(save_dir, 'my_ADV_iter%d.h5'    %    (params['iter_num']-1)  ) )
            print('\033[1;31m ADV iter %d is loaded\033[0m' % (params['iter_num']-1 ))
        if os.path.exists(os.path.join(save_dir, 'my_ave_iter%d.h5'    %    params['iter_num'])):
            ConvAE.vae.load_weights(os.path.join(save_dir, 'my_ave_iter%d.h5'    %    params['iter_num']) )
            print('\033[1;31m ave iter %d is loaded\033[0m' % params['iter_num'] )
        if os.path.exists(os.path.join(save_dir, 'my_discriminator_iter%d.h5'    %    params['iter_num'])):
            ConvAE.Discriminator.load_weights(os.path.join(save_dir, 'my_discriminator_iter%d.h5'    %    params['iter_num']) )
            print('\033[1;31m discriminator iter %d is loaded\033[0m' % params['iter_num'] )
    #ConvAE.vae.load_weights('MNIST_64.h5')
    #ConvAE.Advsior.load_weights('MNIST_ADV_64.h5')


    '''
    sess_config = tf.ConfigProto(device_count={'GPU':0, 'GPU':1})
    session = tf.Session(config=sess_config)
    K.set_session(session)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    '''


    lh = LearningHandler(lr=params['spec_lr'], drop=params['spec_drop'], lr_tensor=ConvAE.learning_rate,
                         patience=params['spec_patience'])





    lh.on_train_begin()
    # one_hots = to_categorical(y_val,10)
    losses_vae = np.empty((zzz,))
    acc = np.empty((zzz,))
    losse = np.empty((zzz,))
    nmi1 = np.empty((zzz,))
    noise = 1*np.random.rand(np.shape(x_train_unlabeled)[0],params['latent_dim'])
    noise1 = 1 * np.random.rand(np.shape(x_train_unlabeled)[0], params['n_clusters'])




    for i in range(zzz):




        x_val_t = ConvAE.encoder.predict(x_val)
        # scale = conv1.get_scale(x_val_y, 1000, params['scale_nbr'])
        x_val_t1 = ConvAE.Advsior.predict(x_val)  #加扰动
        # q= target_distribution(x_val_y)
        x_sp = ConvAE.classfier.predict(x_val_t)
        y_sp = x_sp.argmax(axis=1)
        x_val_y = ConvAE.classfier.predict(x_val_t1+x_val_t)
        y_sp_1 = x_val_y.argmax(axis=1)


        x_val_1 = ConvAE.decoder.predict(x_val_t)
        x_val_2 = ConvAE.decoder.predict(x_val_t1+x_val_t)



        accuracy = print_accuracy(y_sp, y_val, params['n_clusters'])

        if 0==(i+1)%10 and 'defense'==params['training_flag']:
            
            ari=metrics.adjusted_rand_score(y_val, y_sp)

            print('ari:    %f'    %    ari)

        
        nmi1[i] = accuracy
        accuracy = print_accuracy(y_sp_1, y_val, params['n_clusters'])

        if 'defense'==params['training_flag']:
            #ConvAE.fit_defense(x_train_unlabeled,noise,noise1,params['batch_size'])
            losses_vae[i] = ConvAE.train_defense(x_train_unlabeled,noise,noise1,params['batch_size'])  # trainning defense            
        elif 'advsior'==params['training_flag']:
            losses_vae[i] = ConvAE.train_Advsior(x_train_unlabeled,noise,noise1,params['batch_size'])    #training advsior

        
        print("1Z Epoch: {}, loss={:2f},D = removed".format(i, losses_vae[i]))  #remove M
        acc[i] = accuracy

        # nmi1[i] = nmi(y_sp, y_val)
        print('NMI: ' + str(np.round(nmi(y_sp, y_val), 4)))

        print('NMI: ' + str(np.round(nmi(y_sp_1, y_val), 4)))

        if i>1:
            if np.abs(losses_vae[i]-losses_vae[i-1])<0.0001:
                print('STOPPING EARLY')
                break
        

        
    x_val_t = ConvAE.encoder.predict(x_val)

    

    ####
    shuffle_ind=np.random.permutation(  range(np.shape(x_val_t)[0])  )


    x_val_t_permuted=x_val_t[shuffle_ind,:]
    y_sp_permuted=y_sp[shuffle_ind]
    ####

    
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(x_val_t_permuted)
    fig = plt.figure()
    #plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=y_val, cmap=plt.cm.get_cmap("jet", 10))
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=y_sp_permuted, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.savefig(os.path.join(save_dir, 'iter%d_%s_phase.svg'  %  (params['iter_num'], params['training_flag'])) )
    plt.savefig(os.path.join(save_dir, 'iter%d_%s_phase.png'  %  (params['iter_num'], params['training_flag'])) )

    #add UMAP
    umap_2d=UMAP(n_components=2, init='random', random_state=0 )
    proj_2d=umap_2d.fit_transform(x_val_t_permuted)
    fig2=plt.figure()
    #plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=2, c=y_val, cmap=plt.cm.get_cmap("jet", 10))
    plt.scatter(proj_2d[:, 0], proj_2d[:, 1], s=2, c=y_sp_permuted, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.savefig(os.path.join(save_dir, 'iter%d_%s_phase_umap.svg'  %  (params['iter_num'], params['training_flag'])) )
    plt.savefig(os.path.join(save_dir, 'iter%d_%s_phase_umap.png'  %  (params['iter_num'], params['training_flag'])) )


    print("finished training")

    if 'defense'==params['training_flag']:

        labels= y_sp
        write_to_pkl_file=os.path.join(save_dir, 'labels_y_pred_data%d.pkl'  %  params['iter_num'] )
        ftem=open(write_to_pkl_file, 'wb')
        pickle.dump(labels, ftem)
        ftem.close()      
        ConvAE.vae.save_weights(os.path.join(save_dir, 'my_ave_iter%d.h5'    %    params['iter_num']) )
        ConvAE.Discriminator.save_weights(os.path.join(save_dir, 'my_discriminator_iter%d.h5'    %    params['iter_num']))

        '''
        fo=open( os.path.join( save_dir, 'iter%d_defense_result.txt'    %    params['iter_num']), 'w' )
        print_accuracy_to_fo(y_sp, y_val, params['n_clusters'],fo)
        ari=metrics.adjusted_rand_score(y_val, y_sp)
        print('ari:    %f'    %     ari, file=fo)
        score1=metrics.silhouette_score(x_train_original,y_sp)
        score1_ground_truth=metrics.silhouette_score(x_train_original,y_val)
        print('sihouette_score:    %f    ground_truth_sihouette_score:    %f'    %    (score1, score1_ground_truth), file=fo)
        score2=metrics.calinski_harabasz_score( x_train_original,y_sp)
        score2_ground_truth=metrics.calinski_harabasz_score( x_train_original,y_val)
        print('calinski_harabaz_score:    %f    ground_truth_calinski_harabaz_score:    %f'    %    (score2, score2_ground_truth), file=fo)

        try:
            score3=my_dunn_index(x_train_original, labels)
            score3_ground_truth=my_dunn_index(x_train_original, y_val)
            print('dunn_score:    %f    ground_truth_dunn_score:    %f'    %    (score3, score3_ground_truth), file=fo)
        except:
            print('erros in calculating score3!!!')

        score4= metrics.davies_bouldin_score( x_train_original, labels )
        score4_ground_truth= metrics.davies_bouldin_score( x_train_original, y_val )

        print('davies_bouldin_score:    %f    ground_truth_davies_bouldin_score:    %f'    %    (score4, score4_ground_truth), file=fo)

        fo.close()
        '''
    

    elif 'advsior'==params['training_flag']:

        labels= y_sp

        ConvAE.Advsior.save_weights(os.path.join(save_dir, 'my_ADV_iter%d.h5'    %    params['iter_num']) )

        '''
        fo=open( os.path.join( save_dir, 'iter%d_advsior_result.txt'    %    params['iter_num']) ,'w')
        print_accuracy_to_fo(y_sp, y_val, params['n_clusters'],fo)
        ari=metrics.adjusted_rand_score(y_val, y_sp)
        print('ari:    %f'    %     ari, file=fo)
        score1=metrics.silhouette_score(x_train_original,y_sp)
        score1_ground_truth=metrics.silhouette_score(x_train_original,y_val)
        print('sihouette_score:    %f    groudn_truth_sihouette_score:    %f'    %    (score1, score1_ground_truth), file=fo)
        score2=metrics.calinski_harabasz_score( x_train_original,y_sp)
        score2_ground_truth=metrics.calinski_harabasz_score( x_train_original,y_val)
        print('calinski_harabaz_score:    %f    groudn_truth_calinski_harabaz_score:    %f'    %    (score2, score2_ground_truth), file=fo)


        try:
            score3=my_dunn_index(x_train_original, labels)
            score3_ground_truth=my_dunn_index(x_train_original, y_val)
            print('dunn_score:    %f    ground_truth_dunn_score:    %f'    %    (score3, score3_ground_truth), file=fo)
        except:
            print('erros in calculating score3!!!')

        score4= metrics.davies_bouldin_score( x_train_original, labels )
        score4_ground_truth= metrics.davies_bouldin_score( x_train_original, y_val )

        print('davies_bouldin_score:    %f    ground_truth_davies_bouldin_score:    %f'    %    (score4, score4_ground_truth), file=fo)
        fo.close()
        '''



    x_val_y = ConvAE.vae.predict(x_val)[2]
    y_sp = x_val_y.argmax(axis=1)

    print_accuracy(y_sp, y_val, params['n_clusters'])

    nmi_score1 = nmi(y_sp, y_val)
    print('NMI: ' + str(np.round(nmi_score1, 4)))







def target_distribution(q):  # target distribution P which enhances the discrimination of soft label Q
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T






