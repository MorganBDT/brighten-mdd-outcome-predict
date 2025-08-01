#!/bin/bash

mkdir -p results/main/results_Brighten-v1
python clean_mdd_improve_predict_data.py --dest_dir results/main
Rscript miceRanger_imputation.r results/main
python plot_imputation_distributions.py --directory results/main
./parallel_executor.sh forward_selection_commands_main.txt 2
python generate_summary_table.py --input_dir results/main