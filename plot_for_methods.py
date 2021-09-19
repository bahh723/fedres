import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--n_worker", type=int)
parser.add_argument("--K", type=int)
parser.add_argument("--n_round", type=int)


def ema(L, a):
   Q = []
   for i in range(len(L)):
      if i==0: 
        Q.append(L[0]) 
      else:    
        Q.append( (1-a)*Q[-1] + a*L[i] )
   return np.array(Q)

args = parser.parse_args()

scheme_list = [ 'FEDAVG', 'FEDSVRG', 'FEDPROX', 'RESSGD', 'RESAVG', 'RESSVRG', 'RESSIMUL', 'LOCAL']
legend_list = [ 'FedAVG', 'SCAFFOLD', 'FedProx', 'FedResSGD', 'FedResAVG (Opt I)', 'FedResAVG (Opt II)', 'FedResNaive', 'Local']

rates = {}
rates['femnist'] = {
   'FEDAVG': ['0.03', '0'], 
   'FEDSVRG': ['0.03', '0'],
   'RESSGD': ['0.1', '0.003'],
   'RESAVG': ['0.03', '0.003'], 
   'RESSVRG': ['0.03', '0.003'], 
   'RESSIMUL': ['0.03', '0.003'], 
   'LOCAL': ['0.003', '0.003'], 
   'FEDPROX': ['0.01', '0']
}
rates['mnist'] = {
   'FEDAVG': ['0.003', '0'], 
   'FEDSVRG': ['0.003', '0'],
   'RESSGD': ['0.003', '0.003'],
   'RESAVG': ['0.01', '0.003'], 
   'RESSVRG': ['0.01', '0.003'], 
   'RESSIMUL': ['0.03', '0.003'], 
   'LOCAL': ['0.003', '0.01'],
   'FEDPROX': ['0.03', '0']
}

rates['satimage'] = {
   'FEDAVG': ['0.1', '0'], 
   'FEDSVRG': ['0.01', '0'],
   'RESSGD': ['0.1', '0.003'],
   'RESAVG': ['0.03', '0.003'], 
   'RESSVRG': ['0.03', '0.03'], 
   'RESSIMUL': ['0.03', '0.003'], 
   'LOCAL': ['0.03', '0.003'],
   'FEDPROX': ['0.01', '0']
}
rates['sensorless'] = {
   'FEDAVG': ['0.01', '0'], 
   'FEDSVRG': ['0.01', '0'],
   'RESSGD': ['0.003', '0.01'],
   'RESAVG': ['0.003', '0.01'], 
   'RESSVRG': ['0.01', '0.003'], 
   'RESSIMUL': ['0.1', '0.01'], 
   'LOCAL': ['0.003', '0.03'],
   'FEDPROX': ['0.01', '0']
}
rates['letter'] = {
   'FEDAVG': ['0.01', '0'], 
   'FEDSVRG': ['0.01', '0'],
   'RESSGD': ['0.1', '0.0003'],
   'RESAVG': ['0.03', '0.001'], 
   'RESSVRG': ['0.01', '0.0003'], 
   'RESSIMUL': ['0.1', '0.0003'], 
   'LOCAL': ['0.03', '0.001'],
   'FEDPROX': ['0.01', '0']
}

rang = {}
rang['femnist'] = [0.1,0.2]
rang['mnist'] = [0, 0.1]
rang['letter'] = [0.1, 0.3]
rang['sensorless'] = [0.1, 0.3]
rang['satimage'] = [0.05, 0.15]


#scheme_list = [ 'FEDAVG', 'FEDSVRG', 'RESSGD', 'RESAVG', 'RESSVRG', 'RESSIMUL']
#scheme_list = ['RESSGD', 'RESAVG', 'RESSVRG', 'RESSIMUL', 'FEDAVG', 'FEDSVRG', 'FEDPROX', 'LOCAL']
#rate_list = ['0.1', '0.03', '0.01', '0.003']
#if scheme in ['RESSVRG', 'RESAVG', 'RESSGD', 'RESSIMUL', 'LOCAL']: 
#    rate2_list = ['0.03', '0.01', '0.003']
#else:
#    rate2_list = ['0']

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

M = np.zeros((len(scheme_list), n_round))
V = np.zeros((len(scheme_list), n_round))

for i in range(len(scheme_list)):
      s = scheme_list[i]
      nameM = "outtf/" + dataset + "/" + s + "_N" + str(N) + "_R" + str(n_round) + "_K" + str(K) + "_lr" + rates[dataset][s][0] + "_lr2" + rates[dataset][s][1]
      tmpM = np.load(open(nameM, 'rb'))
      M[i, :] = np.mean(tmpM[:,:,1], axis=0) #test loss
      V[i, :] = np.std(tmpM[:,:,1], axis=0)/np.sqrt(tmpM.shape[0])  



#font = {'family' : 'normal', 'size'   : 22}
#matplotlib.rc('font', **font)


plt.figure(figsize=(6,4.5))
rate_legend = []
 
for i in range(len(scheme_list)):
      s = scheme_list[i]
      data_X = np.array(range(50))
      data_Y = ema(M[i,0:50], 0.3)
      data_V = ema(V[i,0:50], 0.3)
      #markers_Y = data_Y[markers_on]
      plt.plot(data_X, data_Y, color=markers_color[i], marker=markers_shape[i], markevery=5+i)     #label='_nolegend_'
      rate_legend.append(s)
      #plt.plot(data_X[markers_on], markers_Y, color=markers_color[k], marker=markers_shape[k], linestyle='None', markersize=11)
      
      plt.fill_between(data_X, data_Y - data_V, data_Y + data_V, color=markers_color[i], alpha=0.3, lw=0)
      #plt.ylim(Lim[i][0], Lim[i][1])
plt.legend(legend_list)
#t = [0, 20, 40, 60, 80, 100]
t = [0, 10, 20, 30, 40, 50]
plt.xticks(t,t)
#plt.xticks(np.arange(0,60,10))
#plt.xticks(fontsize=16, rotation=0)
plt.xlabel('round')
plt.ylabel('average loss')
plt.ylim(rang[dataset][0], rang[dataset][1])
filename="figures/" + dataset + "_N" + str(N) + "_R" + str(n_round) + "_K" + str(K) + ".png"
plt.savefig(filename, bbox_inches = "tight")

