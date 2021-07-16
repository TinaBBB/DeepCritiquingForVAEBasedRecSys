#!/usr/bin/env bash
module load python/3.6
source ~/cevae/bin/activate
cd /home/tinashen/projects/def-ssanner/tinashen/DeepCritiquingForVAEBasedRecSys


python tune_parameters.py --data_dir data/yelp_toronto/ --save_path yelp_rating_final/ce_vae_final_part10.csv --parameters config/yelp/final_performance/ce-vae-part10.yml
