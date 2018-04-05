#!/bin/bash
echo "Starting fcn8 tests..."
echo "Training(1/4): camvid + fcn8 + scratch"
python code/train.py -c code/config/camvid_segmentation_fcn8_scratch_boost.py -e 030418_camvid_fcn8_scratch_boost

echo "Training(2/4): camvid + fcn8 + imagenet(loading imagenet's weights from old test folder)"
python code/train.py -c code/config/camvid_segmentation_fcn8_imagenet_boost.py -e 280318_camvid_fcn8_imagenet_boost

echo "Training(3/4): synthia + fcn8 + scratch"
python code/train.py -c code/config/synthia_segmentation_fcn8_scratch_boost.py -e 030418_synthia_fcn8_scratch_boost

echo "Training(4/4): synthia + fcn8 + imagenet(loading imagenet's weights from old test folder)" 
python code/train.py -c code/config/synthia_segmentation_fcn8_imagenet_boost.py -e 300318_synthia_fcn8_imagenet_boost

