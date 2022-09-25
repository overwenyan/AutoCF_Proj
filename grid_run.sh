#!/bin/bash

if [ "$SHELL" = "/bin/bash" ];then
    echo "your login shell is the bash "
    echo "SHELL is : $SHELL"
else 
    echo "your login shell is not bash but $SHELL"
fi

# fileid=7;
# for arch_assign in [60,70] [70,80] [80,90] [90,100]#[30,40] [40,50] [50,60]#[0,10] [10,20] [20,30]
# do

#     # arch_assign = (( 10*$fileid-10 )) #, 10*$fileid]
#     echo "file_id=$fileid, arch_assign=$arch_assign"
#     # generate_random_hparams
#     python ./main.py  --mode random_single --dataset yelp --data_type implicit --file_id $fileid --arch_assign $arch_assign
#     # (( fileid=$fileid+1 ));
#     fileid=$(($fileid+1))
# done

# dataset='ml-100k'
# dataset='ml-1m'
# dataset='ml-10m'
dataset='amazon-book'
# dataset='yelp'
data_type='implicit'
# data_type = 'explicit'
filename=./main.py
# filename = 'lr_random_nas.py'
# python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 1 --arch_assign [0,10]
# python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 2 --arch_assign [10,20]
# python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 3 --arch_assign [20,30]
# python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 4 --arch_assign [30,40]
python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 5 --arch_assign [40,50]
python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 6 --arch_assign [50,60]
python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 7 --arch_assign [60,70]
python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 8 --arch_assign [70,80]
python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 9 --arch_assign [80,90]
# python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 10 --arch_assign [90,100]
# python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 11 --arch_assign [100,110]
# python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 12 --arch_assign [110,120]
# python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 13 --arch_assign [120,130]
# python $filename  --mode random_single --dataset $dataset --data_type $data_type --file_id 14 --arch_assign [130,135]

# python ./main.py  --mode random_single --dataset $dataset --data_type implicit --file_id  --arch_assign [130,135]