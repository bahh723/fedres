#from util import DelayBuffer, concat_label_feature
import numpy as np
#from vowpalwabbit import pyvw
from time import time
import tensorflow as tf
#tf.disable_v2_behavior() 
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

class Simulator:
    MODE_FEDSGD=0
    MODE_FEDAVG=1
    MODE_FEDSVRG=2
    MODE_RESSGD=3
    MODE_RESSVRG=4
    MODE_LOCAL=5
    MODE_RESSIMUL=6
    MODE_RESSIMULSVRG=7
    MODE_RESAVG=8
    MODE_FEDPROX=9

    def __init__(self, mode=-1, dataset_train=None, dataset_valid=None, dataset_test=None, n_worker=0, lr=0, lr2=0, 
                 n_class=0, mini_batch_size=0, K=0, n_round=0, test_period=1, seed=0):
        self.mode = mode
        self.dataset_train = dataset_train
        self.dataset_valid = dataset_valid
        self.dataset_test = dataset_test
        self.n_worker = n_worker
        self.lr = lr
        self.lr2 = lr2
        self.n_class = n_class
        self.mini_batch_size = mini_batch_size
        self.K = K
        self.n_round = n_round
        self.test_period = test_period
        
        np.random.seed(seed)
        tf.set_random_seed(seed) 
        #self.set_synthesized_parameters()

        self.build_graph()

    def set_synthesized_parameters(self):
        self.dim = 20
        self.n_class = 2
        self.w_gb = np.random.normal(0, 0, size=(self.dim, self.n_class))
        self.w_lc =  np.random.normal(0, 1, size=(self.dim, self.n_class, self.n_worker)) 
        self.noise_level = 0

    def get_batch(self, dataset, batch_size):
        dataset_out = []
        for i in range(self.n_worker):
               idxes = np.random.randint(0, dataset[i]['global'].shape[0], size=(batch_size,) )
               tmp_global = dataset[i]['global'][idxes,:]
               tmp_local = dataset[i]['local'][idxes,:]
               tmp_all = dataset[i]['all'][idxes,:]
               tmp_label = dataset[i]['label'][idxes]
               dataset_out.append( {'all':tmp_all, 'global':tmp_global, 'local': tmp_local, 'label': tmp_label} )
        return dataset_out

    def get_batch_synthesized(self, batch_size): 
        dataset_out = [] 
        #print("batch=", batch_size)
        for i in range(self.n_worker):
            X = np.random.normal(0, 1, (batch_size, self.dim))
            score = np.matmul(X, self.w_gb) + np.matmul(X, self.w_lc[:,:,i]) 
            score += np.random.normal(0, self.noise_level, (batch_size, self.n_class) )
            if self.n_class==1: 
               Y = score[:,0]
            else:
               Y = np.argmax(score, axis=1)
            dataset_out.append({'all': X, 'global': X, 'local': X, 'label': Y})
        return dataset_out 

    def build_graph(self):
        self.dim_global = self.dataset_train[0]['global'].shape[1]   #index: worker idx, all/global/local/label,  
        self.dim_local = self.dataset_train[0]['local'].shape[1]
        dim_global = self.dim_global
        dim_local = self.dim_local
        #======= input place holders =======

        self.X_global = [tf.placeholder(tf.float32, shape=(None, dim_global), name='X_global'+str(i)) for i in range(self.n_worker)]
        self.X_local = [tf.placeholder(tf.float32, shape=(None, dim_local), name='X_local'+str(i)) for i in range(self.n_worker)]
        if self.n_class==1:   # regression problem 
           self.Y = [tf.placeholder(tf.float32, shape=(None, )) for _ in range(self.n_worker)]
        else:
           self.Y = [tf.placeholder(tf.int32, shape=(None, )) for _ in range(self.n_worker)]
        self.X_global_pad = [ tf.pad(self.X_global[i], [[0, 0], [0, 1]], constant_values=1) for i in range(self.n_worker)]
        self.X_local_pad = [ tf.pad(self.X_local[i], [[0, 0], [0, 1]], constant_values=1) for i in range(self.n_worker) ]

        #======= first fully connected layers =======

        dim_H1_global = 32
        dim_H1_local = 32
        self.dim_H1_global = dim_H1_global


        self.W1_global_avg = tf.Variable( tf.random_normal((dim_global + 1, dim_H1_global), 0.5) )
        self.W2_global_avg = tf.Variable( tf.zeros((dim_H1_global, self.n_class)) ) 

        self.W1_global = [tf.Variable( tf.random_normal((dim_global + 1, dim_H1_global), 0.5 )) for _ in range(self.n_worker)]
        self.W2_global = [tf.Variable( tf.zeros((dim_H1_global, self.n_class) )) for _ in range(self.n_worker)]
        self.H1_global = []
        self.H2_global = []

        for i in range(self.n_worker):
            self.H1_global.append( tf.matmul(self.X_global_pad[i], self.W1_global[i]) )
            self.H1_global[i] = tf.nn.relu(self.H1_global[i])
            self.H2_global.append( tf.matmul(self.H1_global[i], self.W2_global[i]) )


        self.W1_local = [tf.Variable( tf.random_normal((dim_local + 1, dim_H1_local),0.5) ) for _ in range(self.n_worker)]
        self.W2_local = [tf.Variable( tf.zeros((dim_H1_local, self.n_class)) ) for _ in range(self.n_worker)]
        self.H1_local = []
        self.H2_local = []
        for i in range(self.n_worker):
            self.H1_local.append( tf.matmul(self.X_local_pad[i], self.W1_local[i]) )
            self.H1_local[i] = tf.nn.relu(self.H1_local[i])
            self.H2_local.append(  tf.matmul(self.H1_local[i], self.W2_local[i]) )

        self.Y_pred = []
        # ========== prediction part for non-linear models ========
        if self.mode in [Simulator.MODE_RESSGD, Simulator.MODE_RESAVG, Simulator.MODE_RESSVRG, Simulator.MODE_RESSIMUL, Simulator.MODE_RESSIMULSVRG, Simulator.MODE_LOCAL]: 
           self.Y_pred = [ self.H2_global[i] + self.H2_local[i] for i in range(self.n_worker)]

        elif self.mode in [Simulator.MODE_FEDSGD, Simulator.MODE_FEDAVG, Simulator.MODE_FEDSVRG, Simulator.MODE_FEDPROX]:
           self.Y_pred = [ self.H2_global[i] for i in range(self.n_worker)]

        #======= calculating the loss =======
        loss_list = []
        for i in range(self.n_worker):
           if self.n_class==1:
              loss_list.append( tf.losses.mean_squared_error(labels=tf.reshape(self.Y[i], [-1]),  predictions=self.Y_pred[i])  )
           else:
              loss_list.append( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.Y[i], [-1]), logits=self.Y_pred[i])  )

        #loss_list = [ tf.reduce_mean(tf.square( self.Y[i]-self.Y_pred[i] )) for i in range(self.n_worker) ]
        self.loss = tf.add_n(loss_list) / self.n_worker

        #======= optimizer (gradient descent) =======

        optimizer_global = tf.train.AdamOptimizer(learning_rate=self.lr)
        optimizer_local = tf.train.AdamOptimizer(learning_rate=self.lr2)

        def subtract_control(tup1, offset):
            tup2 = []
            for i in range(len(tup1)):
               tup2.append( (tup1[i][0] - offset[i], tup1[i][1]) )
            return tup2
         
        self.client_global_op = optimizer_global.minimize(self.loss, var_list=self.W1_global + self.W2_global)    # keeping local parameters fixed
   
        if self.mode in [Simulator.MODE_RESSGD, Simulator.MODE_RESAVG, Simulator.MODE_RESSVRG, Simulator.MODE_RESSIMUL,  Simulator.MODE_LOCAL]: 
            self.client_local_op = optimizer_local.minimize(self.loss, var_list=self.W1_local + self.W2_local)  

        self.gradients = optimizer_global.compute_gradients(self.loss, self.W1_global + self.W2_global) #!!!!!!!!!!!!!!!
        self.control1 = [ tf.placeholder(tf.float32, shape= (dim_global +1, dim_H1_global))  for _ in range(self.n_worker) ]
        self.control2 = [ tf.placeholder(tf.float32, shape= (dim_H1_global, self.n_class)) for  _ in range(self.n_worker)  ]
        
        self.offset_gradients = subtract_control(self.gradients, self.control1 + self.control2)
        self.client_global_control_op = optimizer_global.apply_gradients(self.offset_gradients) # with control variates!! 

        #======= averaging and synchronize among clients =======
        avg_op1 =  tf.assign(self.W1_global_avg , tf.add_n(self.W1_global)/self.n_worker) 
        avg_op2 =  tf.assign(self.W2_global_avg , tf.add_n(self.W2_global)/self.n_worker) 
        sync_op1 = [ tf.assign(self.W1_global[i], self.W1_global_avg) for i in range(self.n_worker) ] 
        sync_op2 = [ tf.assign(self.W2_global[i], self.W2_global_avg) for i in range(self.n_worker) ] 
        self.avg_op = [avg_op1, avg_op2]
        self.sync_op = sync_op1 + sync_op2
        self.init_op = tf.global_variables_initializer()

    def run_simulation(self): 
        sess = tf.Session()
        sess.run(self.init_op)

        te_loss = []
        control1 = []
        control2 = []
        for j in range(self.n_worker):
           control1.append(np.zeros((self.dim_global+1, self.dim_H1_global) ))
           control2.append(np.zeros((self.dim_H1_global, self.n_class) ))

        for r in range(self.n_round): 
           if self.mode==Simulator.MODE_FEDSGD: 
               batch = self.get_batch(self.dataset_train, self.K * self.mini_batch_size)
               feed_dict = {}
               for j in range(self.n_worker):
                   feed_dict[ self.X_global[j] ] = batch[j]['global']
                   feed_dict[ self.X_local[j] ]   = batch[j]['local']
                   feed_dict[ self.Y[j] ] = batch[j]['label']
               sess.run(self.client_global_op, feed_dict=feed_dict)
               sess.run(self.avg_op)
               sess.run(self.sync_op)

           elif self.mode==Simulator.MODE_FEDAVG:
               for _ in range(self.K): 
                   batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                   feed_dict = {}
                   for j in range(self.n_worker):
                       feed_dict[ self.X_global[j] ] = batch[j]['global']
                       feed_dict[ self.X_local[j] ] = batch[j]['local']
                       feed_dict[ self.Y[j] ] = batch[j]['label']
                   sess.run(self.client_global_op, feed_dict=feed_dict)
               sess.run(self.avg_op)
               sess.run(self.sync_op)

           elif self.mode==Simulator.MODE_FEDPROX:
               wbar1 = sess.run(self.W1_global_avg)
               wbar2 = sess.run(self.W2_global_avg)
               for _ in range(self.K): 
                   wcur1 = sess.run(self.W1_global)
                   wcur2 = sess.run(self.W2_global)
                   for j in range(self.n_worker): 
                       control1[j] = - self.lr2 * (wcur1[j] - wbar1)
                       control2[j] = - self.lr2 * (wcur2[j] - wbar2)

                   batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                   feed_dict = {}
                   for j in range(self.n_worker):
                       feed_dict[ self.X_global[j] ] = batch[j]['global']
                       feed_dict[ self.X_local[j] ] = batch[j]['local']
                       feed_dict[ self.Y[j] ] = batch[j]['label']
                       feed_dict[ self.control1[j] ] = control1[j]
                       feed_dict[ self.control2[j] ] = control2[j]
                   sess.run(self.client_global_control_op, feed_dict=feed_dict)


           elif self.mode==Simulator.MODE_FEDSVRG:
               for_next_c1 = []
               for_next_c2 = []
               for _ in range(self.K): 
                   batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                   feed_dict = {}
                   for j in range(self.n_worker):
                       feed_dict[ self.X_global[j] ] = batch[j]['global']
                       feed_dict[ self.X_local[j] ] = batch[j]['local']
                       feed_dict[ self.Y[j] ] = batch[j]['label']
                       feed_dict[ self.control1[j] ] = control1[j]
                       feed_dict[ self.control2[j] ] = control2[j]
                   ### get gradient ###
                   for_next_c1.append( sess.run(self.gradients[0: self.n_worker], feed_dict=feed_dict) )
                   for_next_c2.append( sess.run(self.gradients[self.n_worker: 2*self.n_worker], feed_dict=feed_dict) )
                   sess.run(self.client_global_control_op, feed_dict=feed_dict)
               sess.run(self.avg_op)
               sess.run(self.sync_op)
               control1 = self.calculate_next_c(for_next_c1)
               control2 = self.calculate_next_c(for_next_c2)

           elif self.mode==Simulator.MODE_RESSGD:
               for i in range(self.K):
                   batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                   feed_dict = {}
                   for j in range(self.n_worker):
                       feed_dict[ self.X_global[j] ] = batch[j]['global']
                       feed_dict[ self.X_local[j] ] = batch[j]['local']
                       feed_dict[ self.Y[j] ] = batch[j]['label'] 
                   #print(batch[0]['label'])
                   #train_loss = sess.run(self.loss, feed_dict=feed_dict)
                   #print("train loss=", train_loss)
                   #p = sess.run(self.gradients2[0], feed_dict=feed_dict)
                   #print( p)
                   sess.run(self.client_local_op, feed_dict=feed_dict)
                   
               feed_dict = {}
               batch = self.get_batch(self.dataset_train, self.K * self.mini_batch_size)
               for j in range(self.n_worker): 
                   feed_dict[ self.X_global[j] ] = batch[j]['global']
                   feed_dict[ self.X_local[j] ]   = batch[j]['local']
                   feed_dict[ self.Y[j] ] = batch[j]['label']
               sess.run(self.client_global_op, feed_dict=feed_dict)
               sess.run(self.avg_op)
               sess.run(self.sync_op)


           elif self.mode==Simulator.MODE_RESAVG:
               for i in range(self.K):
                      batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                      feed_dict = {}
                      for j in range(self.n_worker):
                          feed_dict[ self.X_global[j] ] = batch[j]['global']
                          feed_dict[ self.X_local[j] ] = batch[j]['local']
                          feed_dict[ self.Y[j] ] = batch[j]['label']
                      
                      sess.run(self.client_local_op, feed_dict=feed_dict)

               for _ in range(self.K): 
                   batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                   feed_dict = {}
                   for j in range(self.n_worker): 
                       feed_dict[ self.X_global[j] ] = batch[j]['global']
                       feed_dict[ self.X_local[j] ]   = batch[j]['local']
                       feed_dict[ self.Y[j] ] = batch[j]['label']
                   sess.run(self.client_global_op, feed_dict=feed_dict)
               sess.run(self.avg_op)
               sess.run(self.sync_op)


           elif self.mode==Simulator.MODE_RESSVRG:
               for_next_c1 = []
               for_next_c2 = []
               for i in range(self.K):
                   batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                   feed_dict = {}
                   for j in range(self.n_worker):
                       feed_dict[ self.X_global[j] ] = batch[j]['global']
                       feed_dict[ self.X_local[j] ] = batch[j]['local']
                       feed_dict[ self.Y[j] ] = batch[j]['label']
                   sess.run(self.client_local_op, feed_dict=feed_dict)

               for _ in range(self.K): 
                   batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                   feed_dict = {}
                   for j in range(self.n_worker): 
                       feed_dict[ self.X_global[j] ] = batch[j]['global']
                       feed_dict[ self.X_local[j] ]   = batch[j]['local']
                       feed_dict[ self.Y[j] ] = batch[j]['label']
                       feed_dict[ self.control1[j] ] = control1[j]
                       feed_dict[ self.control2[j] ] = control2[j]
                   for_next_c1.append(sess.run(self.gradients[0:self.n_worker], feed_dict=feed_dict))
                   for_next_c2.append(sess.run(self.gradients[self.n_worker: 2*self.n_worker], feed_dict=feed_dict))
                   sess.run(self.client_global_control_op, feed_dict=feed_dict)
               sess.run(self.avg_op)
               sess.run(self.sync_op)
               control1 = self.calculate_next_c(for_next_c1)
               control2 = self.calculate_next_c(for_next_c2)

           elif self.mode==Simulator.MODE_RESSIMUL:
               for _ in range(self.K):
                   batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                   feed_dict = {}
                   for j in range(self.n_worker):
                       feed_dict[ self.X_global[j] ] = batch[j]['global']
                       feed_dict[ self.X_local[j] ] = batch[j]['local']
                       feed_dict[ self.Y[j] ] = batch[j]['label']
                   sess.run(self.client_local_op, feed_dict=feed_dict)

                   batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                   feed_dict = {}
                   for j in range(self.n_worker):
                       feed_dict[ self.X_global[j] ] = batch[j]['global']
                       feed_dict[ self.X_local[j] ] = batch[j]['local']
                       feed_dict[ self.Y[j] ] = batch[j]['label']
                   sess.run(self.client_global_op, feed_dict=feed_dict)
               sess.run(self.avg_op)
               sess.run(self.sync_op)
           elif self.mode==Simulator.MODE_LOCAL: 
               for _ in range(self.K):
                   batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                   feed_dict = {}
                   for j in range(self.n_worker):
                       feed_dict[ self.X_global[j] ] = batch[j]['global']
                       feed_dict[ self.X_local[j] ] = batch[j]['local']
                       feed_dict[ self.Y[j] ] = batch[j]['label']
                   sess.run(self.client_local_op, feed_dict=feed_dict)

                   batch = self.get_batch(self.dataset_train, self.mini_batch_size)
                   feed_dict = {}
                   for j in range(self.n_worker):
                       feed_dict[ self.X_global[j] ] = batch[j]['global']
                       feed_dict[ self.X_local[j] ] = batch[j]['local']
                       feed_dict[ self.Y[j] ] = batch[j]['label']
                   sess.run(self.client_global_op, feed_dict=feed_dict)

           ##### testing #####
           if r % self.test_period==0:

               feed_dict = {}
               for j in range(self.n_worker):
                      feed_dict[ self.X_global[j] ] = self.dataset_test[j]['global']
                      feed_dict[ self.X_local[j] ]   = self.dataset_test[j]['local']
                      feed_dict[ self.Y[j] ] = self.dataset_test[j]['label']
               #err = sess.run(self.loss, feed_dict=feed_dict)
               #test_error = np.mean(err)
               Ypred = sess.run(self.Y_pred, feed_dict=feed_dict)
               err = 0
               total = 0
               for j in range(self.n_worker):
                   Ypred_label = np.argmax(Ypred[j], axis=1)
                   err += np.sum(Ypred_label != self.dataset_test[j]['label']) 
                   total += self.dataset_test[j]['label'].shape[0]
               test_error = err/total

               feed_dict = {}
               for j in range(self.n_worker):
                      feed_dict[ self.X_global[j] ] = self.dataset_valid[j]['global']
                      feed_dict[ self.X_local[j] ]   = self.dataset_valid[j]['local']
                      feed_dict[ self.Y[j] ] = self.dataset_valid[j]['label']
               #err = sess.run(self.loss, feed_dict=feed_dict)
               #test_error = np.mean(err)
               Ypred = sess.run(self.Y_pred, feed_dict=feed_dict)
               err = 0
               total = 0
               for j in range(self.n_worker):
                   Ypred_label = np.argmax(Ypred[j], axis=1)
                   err += np.sum(Ypred_label != self.dataset_valid[j]['label']) 
                   total += self.dataset_valid[j]['label'].shape[0]
               valid_error = err/total




               """
               batch = self.get_batch(self.dataset_test, 256)
               feed_dict = {}
               for j in range(self.n_worker):
                      feed_dict[ self.X_global[j] ] = batch[j]['global']
                      feed_dict[ self.X_local[j] ]   = batch[j]['local']
                      feed_dict[ self.Y[j] ] = batch[j]['label']
               err = sess.run(self.loss, feed_dict=feed_dict)
               valid_error = np.mean(err)
               """
               #Ypred = sess.run(self.Y_pred, feed_dict=feed_dict)
               #print(Ypred[0])
               #print(batch[0]['label'])
               #for j in range(self.n_worker): 
               #     test_error += np.mean(np.square(np.reshape(Ypred[j], [-1]) - batch[j]['label']))
               #test_error /= self.n_worker
               #print("Ypred=", Ypred)
               #print("Y=", Y)
               #print(np.mean(test_error))
                
               te_loss.append([valid_error, test_error])

           if self.mode==Simulator.MODE_FEDPROX: 
               sess.run(self.avg_op)
               sess.run(self.sync_op)

        print("avg test loss: ", np.mean(te_loss))
        te_loss = np.array(te_loss)
        return te_loss       
                
    def calculate_next_c(self, for_next_c):
        
        W_list = [[] for _ in range(self.n_worker)]
        
        assert(len(for_next_c) == self.K)
        assert(len(for_next_c[0]) == self.n_worker)
        assert(len(for_next_c[0][0])==2)
        for k in range(len(for_next_c)):
            for j in range(self.n_worker):
                W_list[j].append(for_next_c[k][j][0])     # gradient

        C = []
        for j in range(self.n_worker):
            C.append(np.mean(W_list[j], 0))  

        C_avg = np.mean(C, 0)

        C_offset = [C[i] - C_avg for i in range(self.n_worker)]
        return C_offset

              
    def test_module(self):     # this function is for debugging purpose
        self.n_worker = 3
        self.K = 1
        self.lr = 0.1
        self.lr2 = 0.1
        n_sample = 10
        feat_dim = 3   
        self.mode = Simulator.MODE_RESSGD    
 
        self.dataset_train = []
        for i in range(self.n_worker):
           inX_all = np.random.rand(n_sample,feat_dim)
           inY = np.random.rand(n_sample,1)
           inX_global = inX_all
           inX_local = inX_all
           self.dataset_train.append({'all':inX_all, 'global':inX_global, 'local':inX_local, 'label':inY})

        self.build_graph()

        sess = tf.Session()
        sess.run(self.init_op)

        ########## test global_op #############
        for i in range(4):
            feed_dict = {}
            #W_global_eval1 = np.reshape(sess.run(self.W_global[0]), [-1])
            #print("")
            #print("round" + str(i) + ":")
            #print("(before) w= ")
            #print(W_global_eval1) 

            for j in range(self.n_worker):
               feed_dict[ self.X_global[j] ] = np.reshape(self.dataset_train[j]['global'][i,:], [1,-1])
               feed_dict[ self.X_local[j] ] = np.reshape(self.dataset_train[j]['local'][i,:], [1,-1])
               feed_dict[ self.Y[j] ] = np.reshape(self.dataset_train[j]['label'][i], [1,-1])
            gradients = sess.run(self.gradients, feed_dict=feed_dict)
            #print(gradients[0])
            #sess.run(self.client_global_op, feed_dict=feed_dict)

            #W_global_eval2 = np.reshape(sess.run(self.W_global[0]), [-1])
            #print("round" + str(i) + ":")
            #print("(after) w= ")
            #print(W_global_eval2)

            #diff = W_global_eval2 - W_global_eval1
            #X_feat = self.dataset_train[0]['global'][i,:]
            #print("diff=", diff)
            #print("X=", X_feat)
            #print("ratio=", np.divide(diff, X_feat))

        ############## test avg and sync #################

        #Ws = []
        #for i in range(self.n_worker):
        #   Ws.append( np.reshape(sess.run(self.b_global[i]), [-1]) )
        #Wmean = np.mean(Ws, 0)
        #print("mean=", Wmean) 

        ## test avg_op
        #avg1 = sess.run(self.b_global_avg)
        #print("before avg:" , avg1)
        #sess.run(self.avg_op)
        #avg2 = sess.run(self.b_global_avg)
        #print("after avg:" , avg2)

        #sess.run(self.sync_op)
        #sync1 = sess.run(self.b_global[0])
        #print("after sync:", sync1)

        ################ test control variates ##################


