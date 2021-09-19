n_worker=50
n_worker_for_split=140
K=50
n_round=60

#dataset=satimage
#mkdir outtf
#mkdir outtf/${dataset}
mkdir outtf

for dataset in femnist
do 

mkdir outtf/${dataset}


for scheme in RESSGD LOCAL RESSIMUL RESAVG RESSVRG FEDPROX FEDAVG FEDSVRG 
do 
   
   if [ "$scheme" = "RESSVRG" ] ; then
      lr_list="0.03"
      lr2_list="0.003" 
   elif  [ "$scheme" = "RESAVG" ] ; then 
      lr_list="0.03"
      lr2_list="0.003"
   elif  [ "$scheme" = "RESSGD" ] ; then
      lr_list="0.1"
      lr2_list="0.003"
   elif  [ "$scheme" = "RESSIMUL" ] ; then 
      lr_list="0.03"
      lr2_list="0.003"
   elif  [ "$scheme" = "LOCAL" ] ; then
      lr_list="0.003"
      lr2_list="0.003"
   elif  [ "$scheme" = "FEDAVG" ] ; then 
      lr_list="0.03"
      lr2_list="0"
   elif  [ "$scheme" = "FEDSVRG" ] ; then
      lr_list="0.03"
      lr2_list="0"
   elif  [ "$scheme" = "FEDPROX" ] ; then
      lr_list="0.01"
      lr2_list="0"
   fi
 
      
   for lr in ${lr_list}
   do
      for lr2 in ${lr2_list}
      do
          echo "$scheme / $lr / ${lr2}"
          
          python3 main.py --dataset femnist_pca100/${dataset} \
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
