import numpy as np

class DelayBuffer:
    def __init__(self, width=0):
        self.buffer = 0.0*np.ones(2*width+1)
        self.width = width
    def set_value(self, time, value):
        self.buffer[time%(2*self.width+1)] = value   
    def get_value(self, time): 
        return self.buffer[time%(2*self.width+1)]

#def extract_label_gb_lc_feature_from_format(vw_format): 
#    format_split = vw_format.split()
#    label = format_split[0]
#    format_x_gb = ""
#    format_x_lc = ""
#    for i in range(2, len(format_split)): 
#        ss = format_split[i].split(":")
#        assert(len(ss)==2)
#        if int(ss[0]<dim_gb):
#           format_x_gb += format_split[i]+" "
#        else:  
#           format_x_lc += format_split[i] + " "
#    return label, format_x_gb.strip(), format_x_lc.strip()
    
def concat_label_feature(label=None, feature=None, base=None): 
    if label==None:
        return "| " + feature
    elif base!=None: 
        return str(label) + " 1 " + str(base) + " | " + feature 
    else: 
        return str(label) + " | " + feature

#def to_vw_example_format(feature, label=None):
#    example_string = ""
#    if label:
#       example_string += "{} | ".format(label)
#    else: 
#       example_string += " | "
#    
#    for i in range(feature.shape[0]): 
#        example_string += "%d:%f " % (i+1, feature[i])
#    return example_string.strip()

import numpy


def output(filename, tr_loss, tr_iter, va_loss, va_iter, te_loss, te_iter): 
    f_output = open(filename, "w+")
    # averaging over workers
    n_worker = tr_loss.shape[0]
    tr_avg_worker = np.zeros((n_worker,))
    va_avg_worker = np.zeros((n_worker,))
    te_avg_worker = np.zeros((n_worker,))
    te_total_risk = 0    
    te_sample = 0

    for j in range(n_worker):
       #print(tr_iter, te_iter) 
       tr_avg_worker[j] = np.mean(tr_loss[j,:int(tr_iter[j])])
       va_avg_worker[j] = np.mean(va_loss[j,:int(va_iter[j])])
       te_avg_worker[j] = np.mean(te_loss[j,:int(te_iter[j])])

       te_risk = [x**2 for x in te_loss[j,:int(te_iter[j])]]
       te_total_risk += np.sum(te_risk)
       te_sample += int(te_iter[j])

    tr_avg = np.mean(tr_avg_worker)
    va_avg = np.mean(va_avg_worker)
    te_avg = np.mean(te_avg_worker)

    print("%6.4f" % (tr_avg), flush=True, file=f_output)
    print("%6.4f" % (va_avg), flush=True, file=f_output)
    print("%6.4f  %8.6f  %d  " % (te_avg, te_total_risk, te_sample), flush=True, file=f_output)
    #print("%6.4f  " % loss_avg, flush=True)
