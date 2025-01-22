# Data analysis code for: Talbot et al. 2024, Comorbid anxiety predicts lower odds of depression improvement during smartphone-delivered psychotherapy

This repository contains the data analysis code for the paper:
Talbot, Morgan B., Omar Costilla-Reyes, and Jessica M. Lipschitz. "Comorbid anxiety symptoms predict lower odds of improvement in depression symptoms during smartphone-delivered psychotherapy." arXiv preprint arXiv:2409.11183 (2024).

## Abstract

Comorbid anxiety disorders are common among patients with major depressive disorder (MDD), and numerous studies have identified an association between comorbid anxiety and resistance to pharmacological depression treatment. However, the impact of anxiety on the effectiveness of non-pharmacological interventions for MDD is not as well understood. In this study, we applied machine learning techniques to predict treatment responses in a large-scale clinical trial (n=493) of individuals with MDD, who were recruited online and randomly assigned to one of three smartphone-based interventions. Our analysis reveals that a baseline GAD-7 questionnaire score in the moderate to severe range (>10) predicts reduced probability of recovery from MDD. Our findings suggest that depressed individuals with comorbid anxiety face lower odds of substantial improvement in the context of smartphone-based therapeutic interventions for depression. Our work highlights a methodology that can identify simple, clinically useful “rules of thumb” for treatment response prediction using interpretable machine learning models.

## Overview of data analysis

The analysis is divided into several steps:

1. Data cleaning and imputation
2. Forward variable selection and model fitting
3. Follow-up decision tree analysis

Follow the step-by-step instructions below to reproduce our results. 

## Data cleaning and imputation

1. Obtain access to the Brighten-V1 dataset from the [Brighten Study Public Researcher Portal](https://www.synapse.org/Synapse:syn10848316/wiki/548727) (you must first complete the online trainings and obtain approval to use the data)
2. Various tables must be downloaded manually as .csv files, placed in a directory called "data," and given the names phq9.csv, baseline_phq9.csv, baseline_demographics.csv, auditc.csv, gad7.csv, mania_psychosis.csv, and sds.csv. 
3. Run the following command (having installed Python with the various dependencies): 
```
python clean_response_predict_data.py
```
Check that a new file "missing_formatted_Brighten-v1_all.csv" has been created

4. Run the following command (having installed R with the [miceRanger](https://cran.r-project.org/web/packages/miceRanger/index.html) package):
```
Rscript miceRanger_imputation.r
```
Check that a new file "miceRanger_imputed_formatted_Brighten-v1_all.csv" has been created

5. To check whether the distributions of imputed variables align with natural values, run the following command: 
```
python plot_imputation_distributions.py
```
This will create new files "binary_vars_distribution.png" and "continuous_vars_distribution.png"

## Forward variable selection and model fitting

All commands for running the forward selection and model fitting are found in forward_selection_commands.txt. 

**Please note:** You can run all of these in parallel using the following: 
```
./parallel_executor.sh forward_selection_commands.txt
```
This will probably take several hours. Results from our execution of this analysis can be found in results_Brighten-v1 (these will be overwritten if you re-run the analysis).

## Follow-up decision tree analysis

You can follow the steps of our follow-up decision tree analysis in decision_tree_threshold_analysis.ipynb