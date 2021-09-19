n_worker=100
K=50
n_round=60

#dataset=satimage
#mkdir outtf
#mkdir outtf/${dataset}
mkdir outtf

for dataset in mnist letter satimage sensorless
do 

mkdir outtf/${dataset}


python3 plot_for_methods.py --dataset ${dataset} \
                   --n_worker ${n_worker} \
                   --K ${K} \
                   --n_round ${n_round}
done
