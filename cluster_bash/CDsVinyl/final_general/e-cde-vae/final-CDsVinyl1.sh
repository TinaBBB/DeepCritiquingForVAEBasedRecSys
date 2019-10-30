#!/usr/bin/env bash
source ~/ENV/bin/activate
cd ~/Dual-Encoder
python reproduce_general_results.py --data_dir data/CDsVinyl/ --tuning_result_path CDsVinyl_rating_tuning --save_path CDsVinyl_final/CDsVinyl_final_result1.csv