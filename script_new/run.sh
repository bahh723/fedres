n_worker=100
n_worker_for_split=100
K=50
n_round=60

#dataset=satimage
#mkdir outtf
#mkdir outtf/${dataset}
mkdir outtf

for dataset in satimage sensorless letter
do 

mkdir outtf/${dataset}


for scheme in RESSGD LOCAL RESSIMUL RESAVG RESSVRG FEDPROX FEDAVG FEDSVRG 
do 
   
   if [ "$scheme" = "RESSVRG" ] || [ "$scheme" = "RESAVG" ] || [ "$scheme" = "RESSGD" ] || [ "$scheme" = "RESSIMUL" ] || [ "$scheme" = "LOCAL" ] ; then
      lr2_list="0.03 0.01 0.003"
      #lr2_list="0.001 0.0003"
   else
      lr2_list="0"
   fi
   lr_list="0.1 0.03 0.01 0.003"
 
      
   for lr in ${lr_list}
   do
      for lr2 in ${lr2_list}
      do
          echo "$scheme / $lr / ${lr2}"
          
          python3 main.py --dataset libsvm/${dataset}/${dataset} \
                      --outname outtf/${dataset}/${scheme}_N${n_worker}_R${n_round}_K${K}_lr${lr}_lr2${lr2} \
                      --scheme ${scheme} \
                      --n_worker ${n_worker} \
                      --n_worker_for_split ${n_worker_for_split} \
                      --mini_batch_size 1 \
                      --K ${K} \
                      --n_round ${n_round} \
                      --lr ${lr} \
                      --lr2 ${lr2}
          
      done
   done
   
   : ' 
   python3 plot_for_single_method.py --dataset ${dataset} \
                      --n_worker ${n_worker} \
                      --K ${K} \
                      --n_round ${n_round} \
                      --scheme ${scheme}
   ' 
done
done
