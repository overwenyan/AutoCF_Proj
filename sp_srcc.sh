#!/bin/bash

if [ "$SHELL" = "/bin/bash" ];then
    echo "your login shell is the bash "
    echo "SHELL is : $SHELL"
else 
    echo "your login shell is not bash but $SHELL"
fi


for dataset in ml-100k
do 
for sp_mode in topk distribute 
do
for data_type in explicit

do
python config_generator.py --sample_portion 0.01 --dataset $dataset --sample_mode $sp_mode --data_type $data_type
python config_generator.py --sample_portion 0.05 --dataset $dataset --sample_mode $sp_mode --data_type $data_type
python config_generator.py --sample_portion 0.1 --dataset $dataset --sample_mode $sp_mode --data_type $data_type
python config_generator.py --sample_portion 0.2 --dataset $dataset --sample_mode $sp_mode --data_type $data_type
python config_generator.py --sample_portion 0.3 --dataset $dataset --sample_mode $sp_mode --data_type $data_type
python config_generator.py --sample_portion 0.4 --dataset $dataset --sample_mode $sp_mode --data_type $data_type

python config_generator.py --sample_portion 0.5 --dataset $dataset --sample_mode $sp_mode --data_type $data_type
python config_generator.py --sample_portion 0.6 --dataset $dataset --sample_mode $sp_mode --data_type $data_type
python config_generator.py --sample_portion 0.7 --dataset $dataset --sample_mode $sp_mode --data_type $data_type
python config_generator.py --sample_portion 0.8 --dataset $dataset --sample_mode $sp_mode --data_type $data_type
python config_generator.py --sample_portion 0.9 --dataset $dataset --sample_mode $sp_mode --data_type $data_type
python config_generator.py --sample_portion 1.0 --dataset $dataset --sample_mode $sp_mode --data_type $data_type
done
done
done