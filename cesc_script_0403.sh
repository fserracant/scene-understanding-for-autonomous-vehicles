#!/bin/bash
############## CESC ###############
echo "Task 5"
#git checkout week2_a5 \
echo "Running A3 from scrath on Belgium Traffic Signs" \
    && python code/train.py -c code/config/btsc_finetune_frz25.py -e belg_finetune_frz25_task5
##################################
