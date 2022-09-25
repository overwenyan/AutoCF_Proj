#!/bin/bash

if [ "$SHELL" = "/bin/bash" ];then
    echo "your login shell is the bash "
    echo "SHELL is : $SHELL"
else 
    echo "your login shell is not bash but $SHELL"
fi

for i in 1 2 3 4 5 6 7, 8, 9
do
python rf_hp_tune.py
done