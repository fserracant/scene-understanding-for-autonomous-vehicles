#!/bin/bash
echo "Testing domain adaptation with Synthia on the Camvid dataset..."
echo "Run (1/4): fcn8 architecture from scratch"
python code/train.py -c code/config/camvid_segmentation_domain_adaptation_fcn8_scratch.py -e 050418_camvid_dom_adapt_fcn8_scratch
echo "Run (2/4): segnet architecture from scratch"
python code/train.py -c code/config/camvid_segmentation_domain_adaptation_segnet_scratch.py -e 050518_camvid_dom_adapt_fcn8_imagenet

echo "Run (3/4): fcn8 architecture from imagenet"
python code/train.py -c code/config/camvid_segmentation_domain_adaptation_fcn8_imagenet.py -e 050518_camvid_dom_adapt_segnet_scratch

echo "Run (4/4): segnet architecture from imagenet"
python code/train.py -c code/config/camvid_segmentation_domain_adaptation_segnet_imagenet.py -e 050518_camvid_dom_adapt_segnet_imagenet

