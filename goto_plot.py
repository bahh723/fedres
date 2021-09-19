import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--n_worker", type=int)
parser.add_argument("--K", type=int)
parser.add_argument("--n_round", type=int)

args = parser.parse_args()

#scheme_list = [ 'FEDAVG', 'FEDSVRG', 'RESSGD', 'RESAVG', 'RESSVRG', 'RESSIMUL']
scheme_list = ['RESSGD', 'RESAVG', 'RESSVRG', 'RESSIMUL', 'FEDAVG', 'FEDSVRG', 'FEDPROX', 'LOCAL']
#scheme_list = ['RESAVG', 'RESSVRG']
scheme_names = ['FedResSGD', 'FedResAVG (Opt I)', 'FedResAVG (Opt II)', 'FedResNaive', 'FedAVG', 'SCAFFOLD', 'FedProx', 'Local']
#scheme_names = [ 'FedResAVG (Opt I)', 'FedResAVG (Opt II)']
#scheme_list = [ 'RESSGD', 'RESAVG']
#markers_on = [0,10, 20, 30, 40, 50]
markers_color = ['tab:red','tab:orange','tab:green', 'tab:blue', 'magenta', 'tab:brown', 'crimson', 'gold']
markers_shape = ['s', 'v','o', 'x', '>', '^', '<', 'D']


N=args.n_worker
K=args.K
dataset = args.dataset
n_round = args.n_round


M = np.zeros((len(scheme_list), n_round))
V = np.zeros((len(scheme_list), n_round))

for i in range(len(scheme_list)):
   nameM = "outtf/" + dataset + "/" + scheme_list[i] + "_N" + str(N) + "_R" + str(n_round) + "_K" + str(K) + "_summaryM"
   nameV = "outtf/" + dataset + "/" + scheme_list[i] + "_N" + str(N) + "_R" + str(n_round) + "_K" + str(K) + "_summaryV"
   tmpM = np.load(open(nameM, 'rb'))
   tmpV = np.load(open(nameV, 'rb'))
   M[i, :] = tmpM
   V[i, :] = tmpV


#font = {'family' : 'normal', 'size'   : 22}
#matplotlib.rc('font', **font)


plt.figure(figsize=(6,4.5)) 
for k in range(len(scheme_list)):
      data_X = np.array(range(n_round))
      data_Y = M[k,:]
      data_V = 10*V[k,:]
      #markers_Y = data_Y[markers_on]
      plt.plot(data_X, data_Y, color=markers_color[k], marker=markers_shape[k], markevery=10+k)     #label='_nolegend_'
      #plt.plot(data_X[markers_on], markers_Y, color=markers_color[k], marker=markers_shape[k], linestyle='None', markersize=11)
      
      #plt.fill_between(data_X, data_Y - data_V, data_Y + data_V, color=markers_color[k], alpha=0.3, lw=0)
      #plt.ylim(Lim[i][0], Lim[i][1])
plt.legend(scheme_names)
#t = [0, 20, 40, 60, 80, 100]
t = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90 ,100]
plt.xticks(t,t)
#plt.xticks(np.arange(0,60,10))
#plt.xticks(fontsize=16, rotation=0)
plt.xlabel('round')
plt.ylabel('average loss')
#plt.ylim(0.04,0.1)
filename=dataset+"_N" + str(N) + "_R" + str(n_round) + "_K" + str(K) + ".png"
plt.savefig(filename, bbox_inches = "tight")




