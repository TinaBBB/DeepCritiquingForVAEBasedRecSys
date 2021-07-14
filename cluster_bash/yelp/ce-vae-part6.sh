#!/usr/bin/env bash
module load python/3.6
source ~/cevae/bin/activate
cd /home/tinashen/projects/def-ssanner/tinashen/DeepCritiquingForVAEBasedRecSys


python tune_parameters.py --data_dir data/yelp_toronto/ --save_path yelp_rating_tuning/ce_vae_tuning_part6.csv --parameters config/yelp/ce-vae-part6.yml
