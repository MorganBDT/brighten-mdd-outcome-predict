#!/bin/bash

# Sensitivity Analysis 1: 12-week PHQ-9 instead of 4-week PHQ-9
mkdir -p results/sens_12wk/results_Brighten-v1
python clean_mdd_improve_predict_data.py --week6plus_phq9 --dest_dir results/sens_12wk
Rscript miceRanger_imputation.r results/sens_12wk week6plus_phq9
python plot_imputation_distributions.py --directory results/sens_12wk
./parallel_executor.sh forward_selection_commands_sens_12wk.txt 2
python generate_summary_table.py --input_dir results/sens_12wk


# Sensitivity Analysis 2: Exclude participants with minimal baseline data
mkdir -p results/sens_exclude_min/results_Brighten-v1
python clean_mdd_improve_predict_data.py --exclude_minimal_baseline_pts --dest_dir results/sens_exclude_min
Rscript miceRanger_imputation.r results/sens_exclude_min
python plot_imputation_distributions.py --week6plus_phq9 --directory results/sens_exclude_min
./parallel_executor.sh forward_selection_commands_sens_exclude_min.txt 2
python generate_summary_table.py --input_dir results/sens_exclude_min