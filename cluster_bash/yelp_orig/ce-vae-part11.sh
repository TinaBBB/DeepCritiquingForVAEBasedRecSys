#!/usr/bin/env bash
module load python/3.6
source ~/cevae/bin/activate
cd /home/tinashen/projects/def-ssanner/tinashen/DeepCritiquingForVAEBasedRecSys


python tune_parameters.py --data_dir data/yelp_orig/ --save_path yelp_orig_rating_tuning/ce_vae_tuning_part11.csv --parameters config/yelp/ce-vae-part11.yml
