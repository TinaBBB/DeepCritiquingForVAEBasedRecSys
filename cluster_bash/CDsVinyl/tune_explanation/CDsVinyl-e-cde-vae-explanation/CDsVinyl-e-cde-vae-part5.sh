#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/Dual-Encoder
python tune_parameters.py --data_dir data/CDsVinyl/ --save_path CDsVinyl_explanation_tuning/e_cdevae_tuning_part5.csv --parameters config/CDsVinyl/e-cde-vae-tune-keyphrase/e-cde-vae-part5.yml --tune_explanation
