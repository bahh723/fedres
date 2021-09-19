import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--n_worker", type=int)
parser.add_argument("--K", type=int)
parser.add_argument("--n_round", type=int)
parser.add_argument("--scheme", type=str)
#parser.add_argument('--prefix', type=str)

args = parser.parse_args()

#scheme_list = [ 'FEDAVG', 'FEDSVRG', 'RESSGD', 'RESAVG', 'RESSVRG', 'RESSIMUL']
#scheme_list = ['RESSGD', 'RESAVG', 'RESSVRG', 'RESSIMUL', 'FEDAVG', 'FEDSVRG', 'FEDPROX', 'LOCAL']
scheme = args.scheme
scheme_name = scheme
rate_list = ['0.1', '0.03', '0.01', '0.003']
if scheme in ['RESSVRG', 'RESAVG', 'RESSGD', 'RESSIMUL', 'LOCAL']: 
    rate2_list = ['0.03', '0.01', '0.003']
else:
    rate2_list = ['0']

#scheme_list = ['RESAVG', 'RESSVRG']
#scheme_names = ['FedResSGD', 'FedResAVG (Opt I)', 'FedResAVG (Opt II)', 'FedResNaive', 'FedAVG', 'SCAFFOLD', 'FedProx', 'Local']
#scheme_names = [ 'FedResAVG (Opt I)', 'FedResAVG (Opt II)']
#scheme_list = [ 'RESSGD', 'RESAVG']
#markers_on = [0,10, 20, 30, 40, 50]
markers_color = ['tab:red','tab:orange','tab:green', 'tab:blue', 'magenta', 'tab:brown', 'crimson', 'gold']*3
markers_shape = ['s', 'v','o', 'x', '>', '^', '<']*3


N=args.n_worker
K=args.K
dataset = args.dataset
n_round = args.n_round

M = np.zeros((len(rate_list), len(rate2_list), n_round))

for i in range(len(rate_list)):
   for j in range(len(rate2_list)): 
      nameM = "outtf/" + dataset + "/" + scheme + "_N" + str(N) + "_R" + str(n_round) + "_K" + str(K) + "_lr" + rate_list[i] + "_lr2" + rate2_list[j]
      tmpM = np.load(open(nameM, 'rb'))
      M[i, j, :] = tmpM[:,1] #test loss


#font = {'family' : 'normal', 'size'   : 22}
#matplotlib.rc('font', **font)


plt.figure(figsize=(6,4.5))
rate_legend = []
 
for i in range(len(rate_list)):
   for j in range(len(rate2_list)):
      k = i*len(rate2_list) + j
      data_X = np.array(range(n_round))
      data_Y = M[i,j,:]
      #markers_Y = data_Y[markers_on]
      plt.plot(data_X, data_Y, color=markers_color[k], marker=markers_shape[k], markevery=10+k)     #label='_nolegend_'
      rate_legend.append('lr='+ rate_list[i] + ", lr2=" + rate2_list[j])
      #plt.plot(data_X[markers_on], markers_Y, color=markers_color[k], marker=markers_shape[k], linestyle='None', markersize=11)
      
      #plt.fill_between(data_X, data_Y - data_V, data_Y + data_V, color=markers_color[k], alpha=0.3, lw=0)
      #plt.ylim(Lim[i][0], Lim[i][1])
plt.legend(rate_legend)
#t = [0, 20, 40, 60, 80, 100]
t = [0, 10, 20, 30, 40, 50, 60]
plt.xticks(t,t)
#plt.xticks(np.arange(0,60,10))
#plt.xticks(fontsize=16, rotation=0)
plt.xlabel('round')
plt.ylabel('average loss')
#plt.ylim(0.04,0.1)
filename="figures/" + dataset + "_" + scheme + "_N" + str(N) + "_R" + str(n_round) + "_K" + str(K) + ".png"
plt.savefig(filename, bbox_inches = "tight")

