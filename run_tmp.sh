n_worker=50
K=50
n_round=60

#mkdir outtf
#mkdir outtf/${dataset}
mkdir outtf

for dataset in femnist satimage sensorless letter
do 

mkdir outtf/${dataset}


python3 plot_for_methods.py --dataset ${dataset} \
                   --n_worker ${n_worker} \
                   --K ${K} \
                   --n_round ${n_round}
done
