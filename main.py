import random
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from time import time
from Simulator import Simulator
from DataGenerator import DataReader

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="")
parser.add_argument("--n_worker", type=int, default=30)
parser.add_argument("--n_worker_for_split", type=int, default=60)
parser.add_argument("--outname", type=str, default="")
parser.add_argument("--scheme", type=str, default="")
parser.add_argument("--lr", type=float)
parser.add_argument("--lr2", type=float, default=0)
parser.add_argument("--dataset", type=str)
parser.add_argument("--n_round", type=int, default=50)
parser.add_argument("--K", type=int, default=0.5)
parser.add_argument("--mini_batch_size", type=int, default=32)


args = parser.parse_args()

n_worker = args.n_worker
n_worker_for_split = args.n_worker_for_split

prefix = time()

data_generator = DataReader(filename=args.dataset, n_worker=n_worker, n_worker_for_split=n_worker_for_split)
#data_generator = None

#tr, va, te= None, None, None
#n_class = 1
tr, va, te, n_class = data_generator.gen_dataset_array()
scheme_list = ['FEDSGD', 'FEDAVG', 'FEDSVRG', 'RESSGD', 'RESSVRG', 'LOCAL', 'RESSIMUL', 'RESSIMULSVRG', 'RESAVG', 'FEDPROX']   
mode = scheme_list.index(args.scheme) 

te_loss = []
for i in range(10):
   print("seed=", i)
   sim = Simulator(    mode=mode, 
                        dataset_train=tr, 
                        dataset_valid=va, 
                        dataset_test=te,  
                        n_worker=n_worker, 
                        mini_batch_size=args.mini_batch_size,
                        K=args.K,
                        lr=args.lr, 
                        lr2=args.lr2,
                        n_class=n_class,
                        n_round=args.n_round,
                        seed=i) 
        
   l = sim.run_simulation()
   te_loss.append(l)

te_loss = np.array(te_loss) 
f = open(args.outname, 'wb')
np.save(f, te_loss) 


