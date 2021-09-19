import numpy as np
import random
import json


class DataReader:
   def __init__(self, filename,  n_worker, n_worker_for_split, scheme=""):
      self.n_worker = n_worker
      self.scheme = scheme   # will be useful in the feature splitting case
      self.filename = filename
      self.n_worker_for_split = n_worker_for_split

      
   def gen_dataset_array(self): 
      n_feat_tr, n_class_tr = self.read_maxf(self.filename + "_tr")
      n_feat_va, n_class_va = self.read_maxf(self.filename + "_va")
      n_feat_te, n_class_te = self.read_maxf(self.filename + "_te")
      n_feat = np.max([n_feat_tr, n_feat_va, n_feat_te])
      n_class = np.max([n_class_tr, n_class_va, n_class_te])

      dataset_tr = self.read_dataset_array(self.filename + "_tr", sparse=False, n_feature=n_feat)
      dataset_va = self.read_dataset_array(self.filename + "_va", sparse=False, n_feature=n_feat)
      dataset_te = self.read_dataset_array(self.filename + "_te", sparse=False, n_feature=n_feat)
      return dataset_tr, dataset_va, dataset_te, n_class

   def read_maxf(self, filename):
      random.seed(self.n_worker) 
      max_f = 0
      min_f = 100000 
      max_label = 0
      min_label = 10000
      order = list(range(self.n_worker_for_split))
      for i in range(self.n_worker):
          fn = filename + "_" + str(self.n_worker_for_split) + "_" + str(order[i])
          with open(fn) as f: 
             for line in f:
                linesp = line.strip().split()
                last_term = linesp[-1]
                last_term = last_term.split(":")
                f_idx = int(last_term[0])

                if f_idx > max_f:
                     max_f = f_idx
                if f_idx < min_f:
                     min_f = f_idx

                label_term = float(linesp[0])
                if label_term > max_label:
                     max_label = label_term
                if label_term < min_label:
                     min_label = label_term
                
     
      assert(min_f >= 0) 
      l_features = list(range(max_f+1))
      #self.global_features = random.sample(l_features, max_f//2)
      self.global_features = l_features      
      self.local_features = l_features 

      return max_f+1, int(max_label-min_label+1)

   def read_dataset_array(self, filename, sparse=False, n_feature=0): 
      dataset = [] 
      random.seed(self.n_worker)
      order = list(range(self.n_worker_for_split))
      random.shuffle(order)


      for i in range(self.n_worker):
         X_global = []
         X_local = []
         X_all = []
         Y = []
         fn = filename + "_" + str(self.n_worker_for_split) + "_" + str(order[i])
         
         f = open(fn)
         for line in f:
            feat_array = np.zeros([n_feature,])  
            line = line.strip()
            linesp = line.split(" ",1)
            label = int(linesp[0])

            feat = linesp[1]
            feat_split = feat.split()

            feat_global = [0 for _ in range(n_feature+10)]
            feat_local = [0 for _ in range(n_feature+10)]
            feat_all = [0 for _ in range(n_feature+10)]

            for entry in feat_split:
               entry_split = entry.split(":")
               idx = int(entry_split[0])
               val = float(entry_split[1])
               if idx in self.global_features:
                   feat_global[idx] = val
               if idx in self.local_features:
                   feat_local[idx] = val
               if idx < n_feature: 
                   feat_all[idx] = val
              

            X_global.append(feat_global)
            X_local.append(feat_local)
            X_all.append(feat_all)
            Y.append(label)

         X_all = np.array(X_all)
         X_global = np.array(X_global)
         X_local = np.array(X_local)
         Y = np.array(Y)
         
         dataset.append({'all':X_all, 'global':X_global, 'local':X_local, 'label':Y})

      return dataset


