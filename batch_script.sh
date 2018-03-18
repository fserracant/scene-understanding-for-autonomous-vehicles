#!/bin/bash
############## JON ###############
#&& git checkout task_a3_resize_vs_crop \
#echo "Task 3" \
#    && echo "Running CROP" \
#    && python code/train.py -c code/config/tt100k_classif_with_crop.py -e 03040000_task3_crop \
#    && echo "Running RESIZE 48x48" \
#    && python code/train.py -c code/config/tt100k_classif_with_resize.py -e 03040000_task3_resize \
#    && echo "Running RESIZE 64x64" \
#    && python code/train.py -c code/config/tt100k_classif_with_resize_64.py -e 03040000_task3_resize_64
##################################
#&& git checkout task_a3_resize_vs_crop \
echo "Task 3" \
    && echo "Running CROP" \
    && python code/train.py -c code/config/tt100k_classif_with_crop.py -e 03021230_task3_crop \
    && echo "Running RESIZE 48x48" \
    && python code/train.py -c code/config/tt100k_classif_with_resize.py -e 03021230_task3_resize \
    && echo "Running RESIZE 64x64" \
    && python code/train.py -c code/config/tt100k_classif_with_resize_64.py -e 030217000_task3_resize_64 \
    && echo "Finished"
