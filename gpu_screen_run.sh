#!/usr/bin/env bash
set -o xtrace

NL=`echo -ne '\015'`

function screen_run {
        local title=$2
        local cmd="$3"

        screen -S $1 -X screen -t $title
        screen -S $1 -p $title -X stuff "$cmd$NL"
}

SCREEN_NAME=screen_run@gaochen

screen -S $SCREEN_NAME -X quit
screen -d -m -S $SCREEN_NAME -t shell -s /bin/bash




# screen_run $SCREEN_NAME c "cd ..; source torch-1.2-py3/bin/activate; cd - ; python ./src/main.py --data_type implicit --mode random_single --dataset yelp2 --use_gpu 1 --mark newbpr --arch_assign $1"
screen_run $SCREEN_NAME c " python ./main.py --data_type implicit --mode random_single --dataset ml-100k --use_gpu 1 --mark newbpr --arch_assign $1"

sleep 1
