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
for opt in Adagrad Adam SGD  # method2
do 
    for lr in 0.001 0.005 0.01 0.05
    do 
        for embedding_dim in 1 2 4 8
        do
            for weight_decay in 1e-6 1e-5 1e-4 1e-3 1e-2
            do
                echo "opt=$opt, lr=$lr, embedding_dim=$embedding_dim, weight_decay=$weight_decay"
                # generate_random_hparams
                python ./main.py  --mode GMF --dataset ml-100k --use_gpu 1 --gpu 7 --device 7 --opt $opt  --lr $lr --embedding_dim $embedding_dim --weight_decay $weight_decay
            done
        done
    done
done
