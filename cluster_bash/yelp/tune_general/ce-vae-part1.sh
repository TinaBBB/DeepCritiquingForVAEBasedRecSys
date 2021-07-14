#!/usr/bin/env bash
source ~/cevae/bin/activate
cd ~/DeepCritiquingForVAEBasedRecSys


python tune_parameters.py --data_dir data/yelp_toronto/ --save_path yelp_rating_tuning/ce_vae_tuning_part1.csv --parameters config/yelp/ce-vae-tune/ce-vae-part1.yml
