#!/bin/bash
#/usr/bin/env bash
set -o xtrace

NL=`echo -ne '\015'`

function screen_run {
        local title=$2
        local cmd="$3"

        screen -S $1 -X screen -t $title
        screen -S $1 -p $title -X stuff "$cmd$NL"
}

SCREEN_NAME=screen_run

screen -S $SCREEN_NAME -X quit
screen -d -m -S $SCREEN_NAME -t shell -s /bin/bash



# screen_run $SCREEN_NAME c "cd ~; source torch-1.2-py3/bin/activate; cd ~/AutoCF_Proj;
# python ./src/main.py --dataset amazon-book --use_gpu 0 --arch_assign $1"
# sleep 1
