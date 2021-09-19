mkdir mnist
mkdir covtype
mkdir letter
mkdir pendigits
mkdir satimage
mkdir sensorless
mkdir usps
mkdir shuttle


python convert.py --file ../libsvm_datasets/mnist --n_worker 100 --ofile ./mnist/mnist 
python convert.py --file ../libsvm_datasets/covtype --n_worker 100 --ofile ./covtype/covtype 
python convert.py --file ../libsvm_datasets/letter --n_worker 100 --ofile ./letter/letter 
python convert.py --file ../libsvm_datasets/pendigits --n_worker 100 --ofile ./pendigits/pendigits 
python convert.py --file ../libsvm_datasets/satimage --n_worker 100 --ofile ./satimage/satimage 
python convert.py --file ../libsvm_datasets/sensorless --n_worker 100 --ofile ./sensorless/sensorless 
python convert.py --file ../libsvm_datasets/usps --n_worker 100 --ofile ./usps/usps 
python convert.py --file ../libsvm_datasets/shuttle --n_worker 100 --ofile ./shuttle/shuttle 
