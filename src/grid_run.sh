#!/bin/bash

if [ "$SHELL" = "/bin/bash" ];then
    echo "your login shell is the bash "
    echo "SHELL is : $SHELL"
else 
    echo "your login shell is not bash but $SHELL"
    # python ./main.py  --mode GMF --dataset ml-100k --use_gpu 1  --opt Adagrad --seed 10 --gpu 7 --device 7
fi

# function generate_random_hparams {
#     local title=$2
#     local cmd="$3"
#     echo "123"
# }
# echo $1
# for opt in Adagrad Adam SGD  # method2
# do 
#     for lr in 0.001 0.005 0.01 0.05
#     do 
#         for embedding_dim in 1 2 4 8
#         do
#             for weight_decay in 1e-6 1e-5 1e-4 1e-3 1e-2
#             do
#                 echo "opt=$opt, lr=$lr, embedding_dim=$embedding_dim, weight_decay=$weight_decay"
#                 # generate_random_hparams
#                 python ./main.py  --mode GMF --dataset ml-100k --use_gpu 1 --gpu 7 --device 7 --opt $opt  --lr $lr --embedding_dim $embedding_dim --weight_decay $weight_decay
#             done
#         done
#     done
# done

for mode in GMF MLP CML SVD
do
for opt in Adagrad Adam  # method2
do 
    for lr in 0.001 0.005 0.01 0.05 0.1 0.5 1.0 1.5 2.0
    do 
    for embedding_dim in 1 2 4 8 16 32 64
    do
        for data_type in explicit implicit
        do
            echo "mode=$mode, opt=$opt, lr=$lr, embedding_dim=$embedding_dim, data_type=$data_type"
            # generate_random_hparams
            python ./main.py  --mode $mode --dataset ml-1m --use_gpu 1 --gpu 6 --device 6 --opt $opt  --lr $lr --embedding_dim $embedding_dim --data_type $data_type
        done
    done
    done
done
done
