import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from keras.layers import *
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from keras import losses
from .layer import stack_layers
from . import costs
from .util import get_scale
import numpy as np
from keras.utils import multi_gpu_model
import keras




class ConvAE:

    def __init__(self,x,params):
        os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu_ids']
        self.gpu=params['gpu']
        self.x = x
        a = tf.shape(self.x)[0]
        self.P = tf.eye(tf.shape(self.x)[0])
        h = x



        filters = params['filters']
        latent_dim = params['latent_dim']
        num_classes = params['n_clusters']
        #self.Dy=Input(shape=(latent_dim,))
        #self.Dy1=Input(shape=(num_classes,))
        self.Dy = tf.placeholder(tf.float32, [None, latent_dim], name='Dy')
        self.Dy1 = tf.placeholder(tf.float32, [None, num_classes], name='Dy1')

        '''
        for i in range(1):
            filters *= 2

            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=2,
                    padding='same')(h)

            h = LeakyReLU(0.2)(h)
            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=1,
                    padding='same')(h)

            h = LeakyReLU(0.2)(h)

        for i in range(1):
            filters *= 2
            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=2,
                    padding='same')(h)
            h = LeakyReLU(0.2)(h)
            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=1,
                    padding='same')(h)
            h = LeakyReLU(0.2)(h)
        '''

        #change the above 2D network to 1D network
        #x to z network
        #using Dense

        h=Dense(200)(h)
        h=BatchNormalization()(h)  #add Batchnorm
        h=Activation('relu')(h)
        h=Dense(100)(h)
        h=BatchNormalization()(h)  #add Batchnorm
        h=Activation('relu')(h)


        #h_shape = K.int_shape(h)[1:]
        h_shape = K.int_shape(h)[1]
        h = Flatten()(h)

        z_mean = Dense(latent_dim)(h) # p(z|x)的均值  add relu
        z_mean = BatchNormalization()(z_mean)    #add Batchnorm
        z_mean = Activation('relu')(z_mean)
        z_log_var = Dense(latent_dim)(h) # p(z|x)的方差 add relu
        z_log_var = BatchNormalization()(z_log_var)
        z_log_var = Activation('relu')(z_log_var)


# adversarial network

        z = x
        z = Flatten()(z)
        z = Dense(200,name='a1')(z) #add relu
        z = BatchNormalization()(z)    #add Batchnorm
        z = Activation('relu')(z)
        z = Dense(100,name='a2')(z)  #add relu
        z = BatchNormalization()(z)    #add Batchnorm
        z = Activation('relu')(z)
        z = Dense(latent_dim,name='a3')(z)  #add relu
        z = BatchNormalization()(z)    #add Batchnorm
        z = Activation('relu')(z)

        self.Advsior = Model(x,z)

        pertation = self.Advsior(x)

#Decoder



        z = Input(shape=(latent_dim,))
        h = z
        h = Dense(np.prod(h_shape))(h)    #no numpy inmported  add relu
        h = BatchNormalization()(h)    #add Batchnorm
        h = Activation('relu')(h)
        #h = Reshape( h_shape )(h)
        #h = Flatten()(h)
        '''
        for i in range(2):
            h = Conv2DTranspose(filters=filters,
                                kernel_size=3,
                                strides=1,
                                padding='same')(h)
            h = LeakyReLU(0.2)(h)
            h = Conv2DTranspose(filters=filters,
                                kernel_size=3,
                                strides=2,
                                padding='same')(h)
            h = LeakyReLU(0.2)(h)
            filters //= 2

        x_recon = Conv2DTranspose(filters=1,
                                kernel_size=3,
                                activation='sigmoid',
                                padding='same')(h)
        '''

        h=Dense(100)(h) 
        h = BatchNormalization()(h)    #add Batchnorm
        h = Activation('relu')(h)
        
        x_recon=Dense(200)(h)  
        h = BatchNormalization()(h)    #add Batchnorm
        h = Activation('relu')(h)

        x_recon=Dense(params['feather_len'])(h)  
        h = BatchNormalization()(h)    #add Batchnorm
        h = Activation('relu')(h)

        self.decoder = Model(z, x_recon)



#clustering layer
        z = Input(shape=(latent_dim,))
        y = Dense(1024, name='c1')(z)
        y = BatchNormalization()(y)    #add Batchnorm
        y = Activation('relu')(y)
        # y = Lambda(GCN)(y)
        y = Dense(1024,name='c2')(y)
        y = BatchNormalization()(y)    #add Batchnorm
        y = Activation('relu')(y)
        # y = Lambda(GCN)(y)
        y = Dense(512,name='c3')(y)
        y = BatchNormalization()(y)    #add Batchnorm
        y = Activation('relu')(y)
        # y = Lambda(GCN)(y)
        y = Dense(num_classes, activation='softmax')(y)

        self.classfier = Model(z, y)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
            return [z_mean + K.exp(z_log_var / 2) * epsilon,K.exp(z_log_var / 2) * epsilon]

        #z_mean_1 = z_mean + pertation
        z_mean_1 = z_mean + 0.1 * pertation

        z,resample = Lambda(sampling)([z_mean, z_log_var])
        z_1 = z_mean_1+resample
        self.encoder = Model(x, z_mean)

        x_recon = self.decoder(z)
        x_recon1 = self.decoder(z_1)

        y = self.classfier(z_mean)
        y_1 = self.classfier(z_mean_1)

        gaussian = Gaussian(num_classes)
        z_prior_mean = gaussian(z)

        self.vae = Model(x, [x_recon,z_prior_mean,y])


# graph module

        W = costs.knn_affinity(z_mean, params['n_nbrs'], scale=1.97, scale_nbr=params['scale_nbr'])
        W = W - self.P
        layers = [
                  {'type': 'Orthonorm', 'name':'orthonorm'}
                  ]

        outputs = stack_layers(y,layers)
        Dy = costs.squared_distance(outputs)

        loss_SPNet =1* (K.sum(W * Dy))

# MIE
        def shuffling(x):
            idxs = K.arange(0, K.shape(x)[0])
            idxs = tf.random_shuffle(idxs)  #no attribute 'tf' in module 'keras.backend'

            return K.gather(x, idxs)

        z_shuffle = Lambda(shuffling)(z_mean)
        z_z_1 = Concatenate()([z_mean, z_mean])
        z_z_2 = Concatenate()([z_mean, z_shuffle])

        z_in = Input(shape=(latent_dim * 2,))
        z1 = z_in
        z1 = Dense(latent_dim)(z1)
        z1 = BatchNormalization()(z1)    #add Batchnorm
        z1 = Activation('relu')(z1)

        z1 = Dense(latent_dim)(z1)
        z1 = BatchNormalization()(z1)    #add Batchnorm
        z1 = Activation('relu')(z1)

        z1 = Dense(latent_dim)(z1)
        z1 = BatchNormalization()(z1)    #add Batchnorm
        z1 = Activation('relu')(z1)
        
        z1 = Dense(1, activation='sigmoid')(z1)

        GlobalDiscriminator = Model(z_in, z1)

        z_z_1_scores = GlobalDiscriminator(z_z_1)
        z_z_2_scores = GlobalDiscriminator(z_z_2)
        global_info_loss = - K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))


#clustering network


        z_log_var = K.expand_dims(z_log_var, 1)

        lamb = 2  # 这是重构误差的权重，它的相反数就是重构方差，越大意味着方差越小。
        xent_loss = 1 * K.mean((x - x_recon[...,np.newaxis]) ** 2, 0)

        kl_loss = - 0.5 * (1 + z_log_var - K.square(K.expand_dims(z_mean, 1) - z_prior_mean) - K.exp(z_log_var))
        # kl_loss = - 0.5 * (z_log_var - K.square(z_prior_mean))
        kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0)
        cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)

        #self.module_loss = 0.01 * lamb * K.sum(xent_loss)+1*K.sum(kl_loss)+1*K.sum(cat_loss)
        self.module_loss = lamb * K.sum(xent_loss)+1*K.sum(kl_loss)+1*K.sum(cat_loss)

# Attack loss


        selfloss = 1 * K.mean((z_mean_1 - z_mean) ** 2, 0)

        Dis = tf.diag_part(tf.matmul(y,y_1,transpose_b=True))

        xent1_loss = 1 * K.mean((x_recon - x_recon1) ** 2, 0)
        #self.adv_loss = lamb * K.sum(xent1_loss) + 0.04 * K.sum(Dis) + 1 * K.sum(selfloss)
        #self.adv_loss = 0.01 * lamb * K.sum(xent1_loss) + 0.04 * K.sum(Dis) + 0.1 * K.sum(selfloss)    
        #self.adv_loss = lamb * K.sum(xent1_loss) + 0.04 * K.sum(Dis) + 1 * K.sum(selfloss)    
        self.adv_loss = lamb * K.sum(xent1_loss) + 0.01 * K.sum(Dis) + 1 * K.sum(selfloss)

#defense


        #z_in = Input(shape=(858,))
        z_in = Input(shape=(params['feather_len']+params['latent_dim']+params['n_clusters'],))   # feather_len + latent_dim + cluster_num
        z1 = z_in
        z1 = Dense(latent_dim, activation='relu')(z1)
        z1 = Dense(latent_dim, activation='relu')(z1)
        z1 = Dense(1, activation='sigmoid')(z1)

        self.Discriminator = Model(z_in, z1)


        c1 = tf.concat([tf.reshape(x_recon,[-1,params['feather_len']]),y],1)    # feather_len
        c2 = tf.concat([tf.reshape(x_recon1,[-1,params['feather_len']]),y_1],1)    #feather_len


        c1_shuffle = Lambda(shuffling)(c1)
        z_z_1 = Concatenate()([z_mean_1, c1])
        z_z_2 = Concatenate()([z_mean_1, c1_shuffle])

        z_z_1_scores = self.Discriminator(z_z_1)
        z_z_2_scores = self.Discriminator(z_z_2)
        info_loss_c1 = - K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))


        c2_shuffle = Lambda(shuffling)(c2)
        z_z_1 = Concatenate()([z_mean, c2])
        z_z_2 = Concatenate()([z_mean, c2_shuffle])

        z_z_1_scores = self.Discriminator(z_z_1)
        z_z_2_scores = self.Discriminator(z_z_2)
        info_loss_c2 = - K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))



        self.loss_defense = self.module_loss+2.5*(K.sum(info_loss_c1)+K.sum(info_loss_c2))





        self.D = K.mean(K.sum(tf.abs(z_mean - z_mean_1), 0))



        self.learning_rate = tf.Variable(0., name='spectral_net_learning_rate')
        self.train_step1 = tf.train.AdamOptimizer().minimize(self.loss_defense,var_list=[self.vae.weights,self.Discriminator.weights])  # trainning step1
        self.train_step2 = tf.train.AdamOptimizer().minimize(self.adv_loss, var_list=self.Advsior.weights)  # trainning step2

        self.train_step_initial=tf.train.AdamOptimizer().minimize(self.module_loss,var_list=[self.vae.weights])
        self.train_step_initial2 = tf.train.AdamOptimizer().minimize(self.loss_defense,var_list=[self.Discriminator.weights])


        
        '''
        self.defense_model= Model([x,self.Dy,self.Dy1] , [x_recon,z_prior_mean,y])

        #self.train_step1_keras =keras.optimizers.Adam().minimize(self.loss_defense, var_list=[self.vae.weights,self.Discriminator.weights] )  # trainning step1

        self.train_step1_keras=tf.keras.optimizers.Adam()
        self.train_step1_keras.minimize(  self.loss_defense,var_list=[self.vae.weights,self.Discriminator.weights] )

        #self.train_step1_keras =tf.keras.optimizers.Adam().minimize(self.loss_defense, var_list=[self.vae.weights,self.Discriminator.weights] )
        #self.train_step1_keras =tf.compat.v1.keras.optimizers.Adam().minimize(self.loss_defense,var_list=self.defense_model.weights)  # trainning step1
        self.train_step2_keras =keras.optimizers.Adam().minimize(self.adv_loss, var_list=self.Advsior.weights)  # trainning step2
        '''



        #self.pre_train_vae_step1=tf.train.AdamOptimizer().minimize(self.module_loss,var_list=self.vae.weights)  #added
        #self.pre_train_advsior_step2=tf.train.AdamOptimizer().minimize(self.adv_loss, var_list=self.Advsior.weights)  #added
        K.get_session().run(tf.variables_initializer(self.vae.trainable_weights))


    def train_Advsior(self, x_train_unlabeled,x_dy,x_dy1,batch_size):
        # create handler for early stopping and learning rate scheduling

        losses = self.train_vae_step(
                return_var=[self.adv_loss],
                updates=[self.train_step2]+self.Advsior.updates,
                x_unlabeled=x_train_unlabeled,
                inputs=self.x,
                x_dy=x_dy,
                x_dy1=x_dy1,
                batch_sizes=batch_size,
                batches_per_epoch=50)





        return losses


    def train_defense(self, x_train_unlabeled,x_dy,x_dy1,batch_size):
        # create handler for early stopping and learning rate scheduling

        losses = self.train_vae_step(
                return_var=[self.loss_defense],
                updates=[self.train_step1]+self.vae.updates+self.Discriminator.updates,
                x_unlabeled=x_train_unlabeled,
                inputs=self.x,
                x_dy=x_dy,
                x_dy1=x_dy1,
                batch_sizes=batch_size,
                batches_per_epoch=50)
        return losses
    
    def train_initial(self, x_train_unlabeled,x_dy,x_dy1,batch_size):
        # create handler for early stopping and learning rate scheduling

        losses = self.train_vae_step(
                return_var=[self.module_loss],
                updates=[self.train_step_initial]+self.vae.updates,
                x_unlabeled=x_train_unlabeled,
                inputs=self.x,
                x_dy=x_dy,
                x_dy1=x_dy1,
                batch_sizes=batch_size,
                batches_per_epoch=50)
        return losses

    def train_initial2(self, x_train_unlabeled,x_dy,x_dy1,batch_size):
        # create handler for early stopping and learning rate scheduling

        losses = self.train_vae_step(
                return_var=[self.loss_defense],
                updates=[self.train_step_initial2]+self.Discriminator.updates,
                x_unlabeled=x_train_unlabeled,
                inputs=self.x,
                x_dy=x_dy,
                x_dy1=x_dy1,
                batch_sizes=batch_size,
                batches_per_epoch=50)
        return losses
    
    
    '''
    def fit_defense(self, x_train_unlabeled,x_dy,x_dy1,batch_size):
        defense_model_parrarel=multi_gpu_model(self.defense_model, gpus = self.gpu)
        defense_model_parrarel.compile(optimizer=self.train_step1_keras)
        defense_model_parrarel.fit(x=[ x_train_unlabeled,x_dy,x_dy1], batch_size=batch_size, steps_per_epoch=100 )
        '''




    def train_vae_step(self,return_var, updates, x_unlabeled, inputs,x_dy,x_dy1,
                   batch_sizes,
                   batches_per_epoch=100):

        return_vars_ = np.zeros(shape=(len(return_var)))

        # scale = get_scale(x_dy, 1000, 2)
        # train batches_per_epoch batches
        for batch_num in range(0, batches_per_epoch):
            feed_dict = {K.learning_phase(): 1}

            # feed corresponding input for each input_type

            batch_ids = np.random.choice(len(x_unlabeled), size=batch_sizes, replace=False)
            feed_dict[inputs] = x_unlabeled[batch_ids]
            feed_dict[self.Dy]=x_dy[batch_ids]
            feed_dict[self.Dy1] = x_dy1[batch_ids]


                        # feed_dict[P]=P[batch_ids]

            all_vars = return_var + updates
            all_vars_R =  K.get_session().run((all_vars), feed_dict=feed_dict)

            return_vars_ += np.asarray(all_vars_R[:len(return_var)])


            # M = K.get_session().run((self.centers_d),feed_dict=feed_dict)
            # print('confusion matrix{}: '.format(''))
            # print(np.round(centers_d, 2))
            # print(np.round(D, 4))
        return return_vars_



class Gaussian(Layer):

    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)
    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean',
                                    shape=(self.num_classes, latent_dim),
                                    initializer='zeros')
    def call(self, inputs):
        z = inputs # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z * 0 + K.expand_dims(self.mean, 0)
    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])
