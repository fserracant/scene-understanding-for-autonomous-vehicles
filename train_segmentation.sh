#!/bin/bash
echo "Starting fine-tuning for dataset: 'synthia'..."
echo "Training(1/4): synthia + fcn8 + scratch"
python code/train.py -c code/config/synthia_segmentation_fcn8_scratch_boost.py -e 300318_synthia_fcn8_scratch_boost

echo "Training(2/4): synthia + segnet + scratch"
python code/train.py -c code/config/synthia_segmentation_segnet_scratch_boost.py -e 300318_synthia_segnet_scratch_boost

echo "Training(3/4): synthia + fcn8 + imagenet" 
python code/train.py -c code/config/synthia_segmentation_fcn8_imagenet_boost.py -e 300318_synthia_fcn8_imagenet_boost

echo "Training(4/4): synthia + segnet + imagenet"
python code/train.py -c code/config/synthia_segmentation_segnet_imagenet_boost.py -e 300318_synthia_segnet_imagenet_boost
