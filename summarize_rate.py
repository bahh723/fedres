import numpy as np
import glob
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--scheme", type=str, default="")
parser.add_argument("--n_worker", type=int, default=0)
parser.add_argument("--K", type=int, default=0)
parser.add_argument("--n_round", type=int, default=0)


args = parser.parse_args()

K = args.K
dataset = args.dataset
N = args.n_worker
scheme = args.scheme
R = args.n_round

pre = "outtf/" + dataset + "/" + scheme + "_N" + str(N) + "_R" + str(R) + "_K" + str(K)
list_with_pre = glob.glob(pre)
#print("list with pre", list_with_pre)


list_with_lr = []
for f in list_with_pre:
   fsp = f.split("_seed") 
   list_with_lr.append(fsp[0])
list_with_lr = list(set(list_with_lr))

#print("list with lr", list_with_lr)

B_va = {}
B_te = {}
Mean_va = {}
Mean_te = {}
Var_te = {}
Best_f = None

for f in list_with_lr:
   list_with_seed = glob.glob(f + "_*seed*")
   B_va[f] = []
   B_te[f] = []
   print("f=", f)
   print("list_with_seed=", list_with_seed)
   for g in range(len(list_with_seed)): 
      tmp = np.load(open(list_with_seed[g], 'rb'))
      #print(tmp.shape) 
      n_round = tmp.shape[0]
      #print(n_round)
      B_va[f].append(tmp[0: n_round, 0])   # only consider the last half 
      B_te[f].append(tmp[:, 1])                     # tmp's size: (n_round, 2)
   Mean_va[f] = np.mean(B_va[f], axis=0)
   Mean_te[f] = np.mean(B_te[f], axis=0)
   Var_te[f] = np.std(B_te[f], axis=0) / np.sqrt(len(list_with_seed))
   if Best_f == None:
      Best_f = f
   elif np.mean(Mean_va[f]) < np.mean(Mean_va[Best_f]): 
      Best_f = f

np.save(open(pre + "_summaryM", 'wb'), Mean_te[Best_f])
np.save(open(pre + "_summaryV", 'wb'), Var_te[Best_f])
print(Best_f, file=open(pre + "_summary", 'w'))   

