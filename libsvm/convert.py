import argparse
import random
parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)
parser.add_argument("--n_worker", type=int)
parser.add_argument("--ofile", type=str)
args = parser.parse_args()
n_worker = args.n_worker


ofile = args.ofile
random.seed(0)

#### collect the original dataset ####
f = open(args.file)
dataset_ori = {}
dataset_ori_te = {}
dataset_ori_va = {}
c = 0
for line in f: 
    linesp = line.split()
    clas = int(linesp[0])
    if clas not in dataset_ori: 
        dataset_ori[clas] = [line]
    else:
        dataset_ori[clas].append(line)
    c+=1

for clas in dataset_ori: 
    l = len(dataset_ori[clas])
    random.shuffle(dataset_ori[clas])
    dataset_ori_va[clas] = dataset_ori[clas][int(0.7*l):int(0.85*l)]
    dataset_ori_te[clas] = dataset_ori[clas][int(0.85*l):]
    dataset_ori[clas] = dataset_ori[clas][:int(0.7*l)]

class_count = {}
c = 0
for clas in dataset_ori: 
    class_count[clas] = len(dataset_ori[clas])


n_samp_per = 5

dataset_txt = [[] for _ in range(n_worker)]
dataset_txt_va = [[] for _ in range(n_worker)]
dataset_txt_te = [[] for _ in range(n_worker)]

n_class = len(dataset_ori)
sample_counter = {}
for k in class_count: 
     sample_counter[k] = 0

sels = [[] for _ in range(n_worker)]
for i in range(n_worker): 
    sel = random.sample(class_count.keys(), 2)
    while class_count[sel[0]] - sample_counter[sel[0]] < n_samp_per or class_count[sel[1]] - sample_counter[sel[1]] < n_samp_per: 
       sel = random.sample(class_count.keys(), 2)
    if sel[0] > sel[1]: 
       sel = [sel[1], sel[0]]

    sels[i] = sel
    range0 = list(range(sample_counter[sel[0]], sample_counter[sel[0]] + n_samp_per))
    range1 = list(range(sample_counter[sel[1]], sample_counter[sel[1]] + n_samp_per))
    dataset_txt[i] += [ dataset_ori[sel[0]][j] for j in range0 ]
    dataset_txt[i] += [ dataset_ori[sel[1]][j] for j in range1 ]
    sample_counter[sel[0]] += n_samp_per
    sample_counter[sel[1]] += n_samp_per

    rva0 = random.choices(dataset_ori_va[sel[0]], k=n_samp_per)
    rva1 = random.choices(dataset_ori_va[sel[1]], k=n_samp_per)
    dataset_txt_va[i] += rva0 + rva1
    rte0 = random.choices(dataset_ori_te[sel[0]], k=n_samp_per)
    rte1 = random.choices(dataset_ori_te[sel[1]], k=n_samp_per)
    dataset_txt_te[i] += rte0 + rte1

    random.shuffle(dataset_txt[i])
    random.shuffle(dataset_txt_va[i])
    random.shuffle(dataset_txt_te[i])
    # ==========================================================================================

#print(len(dataset_txt[0]))


def write_to(f, i, txt):
    for j in range(len(txt)): 
        line = txt[j].strip()
        linesp = line.split(" ", 1)
        if int(linesp[0]) == sels[i][0]: 
            label = 0
        elif int(linesp[0]) == sels[i][1]:
            label = 1
        else:
            assert(False)

        line = str(label) + " " + linesp[1]
        print(line, file=f)        


#dataset_raw = [[] for _ in range(n_worker)]
for i in range(n_worker):
    with open(ofile + "_tr_"+str(n_worker) + "_" + str(i), 'w') as f:
       write_to(f, i, dataset_txt[i])
    
    with open(ofile + "_te_"+str(n_worker) + "_" + str(i), 'w') as f: 
       write_to(f, i, dataset_txt_te[i])
    
    with open(ofile + "_va_"+str(n_worker) + "_" + str(i), 'w') as f: 
       write_to(f, i, dataset_txt_va[i])
       
