#!/usr/bin/env bash
sbatch --nodes=1 --time=00:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner ce-vae-part1.sh
sbatch --nodes=1 --time=00:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner ce-vae-part2.sh
sbatch --nodes=1 --time=00:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner ce-vae-part3.sh
sbatch --nodes=1 --time=00:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner ce-vae-part4.sh
sbatch --nodes=1 --time=00:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner ce-vae-part5.sh
sbatch --nodes=1 --time=00:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner ce-vae-part6.sh
sbatch --nodes=1 --time=00:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner ce-vae-part7.sh
sbatch --nodes=1 --time=00:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner ce-vae-part8.sh
sbatch --nodes=1 --time=00:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner ce-vae-part9.sh
sbatch --nodes=1 --time=00:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=def-ssanner ce-vae-part10.sh

