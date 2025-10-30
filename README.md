# Data analysis code: Comorbid anxiety predicts lower odds of MDD improvement in a trial of smartphone-delivered interventions

This repository contains the data analysis code for the paper:

Talbot, Morgan B., Jessica M. Lipschitz*, and Omar Costilla-Reyes*. "Comorbid anxiety predicts lower odds of MDD improvement in a trial of smartphone-delivered interventions." _Journal of Affective Disorders_ 394 (2026): 120416. *Co-Senior Authors. DOI: [10.1016/j.jad.2025.120416](https://doi.org/10.1016/j.jad.2025.120416)

[arXiv Link](https://arxiv.org/abs/2409.11183)

```bibtex
@article{talbot2026comorbid,
  title = {Comorbid anxiety predicts lower odds of MDD improvement in a trial of smartphone-delivered interventions},
  journal = {Journal of Affective Disorders},
  volume = {394},
  pages = {120416},
  year = {2026},
  issn = {0165-0327},
  doi = {https://doi.org/10.1016/j.jad.2025.120416},
  url = {https://www.sciencedirect.com/science/article/pii/S0165032725018580},
  author = {Morgan B. Talbot and Jessica M. Lipschitz and Omar Costilla-Reyes},
  note = {Jessica M. Lipschitz and Omar Costilla-Reyes are co-senior authors.}
}
```

## Abstract

Comorbid anxiety disorders are common among patients with major depressive disorder (MDD), but their impact on outcomes of digital and smartphone-delivered interventions is not well understood. This study is a secondary analysis of a randomized controlled effectiveness trial (n=638) that assessed three smartphone-delivered interventions: Project EVO (a cognitive training app), iPST (a problem-solving therapy app), and Health Tips (an active control). We applied classical machine learning models (logistic regression, support vector machines, decision trees, random forests, and k-nearest-neighbors) to identify baseline predictors of MDD improvement at 4 weeks after trial enrollment. Our analysis produced a decision tree model indicating that a baseline GAD-7 questionnaire score of 11 or higher, a threshold consistent with at least moderate anxiety, strongly predicts lower odds of MDD improvement in this trial. Our exploratory findings suggest that depressed individuals with comorbid anxiety have reduced odds of substantial improvement in the context of smartphone-delivered interventions, as the association was observed across all three intervention groups. Our work highlights a methodology that can identify interpretable clinical thresholds, which, if validated, could predict symptom trajectories and inform treatment selection and intensity.


## Overview of data analysis

The analysis is divided into several steps:

1. Data cleaning and imputation
2. Forward variable selection and model fitting
3. Follow-up decision tree analysis
4. Sensitivity analyses

Follow the step-by-step instructions below to reproduce our results. 

For best results, you can reproduce the Python and R environments we used for our analyses. You can recreate our Anaconda environment using: 

```
conda env create -f environment.yml
conda activate brighten_mdd_outcome_predict
```

You can also recreate our R environment on Ubuntu/Debian. "rig" is a tool that can install specific versions of R - note that the following will install R from scratch and set the default R version to 4.4.1 on your machine. 

```
`which sudo` sh -c 'echo "deb http://rig.r-pkg.org/deb rig main" > /etc/apt/sources.list.d/rig.list'
`which sudo` apt update
`which sudo` apt install r-rig
rig add 4.4.1
rig default 4.4.1

# Might or might not be needed: install curl
sudo apt install libcurl4-openssl-dev
```

To recreate the virtual "renv" environment (after running "R" to enter the R command line):

```
if (!require("renv")) install.packages("renv")
renv::restore()
```


## Data cleaning and imputation

1. Obtain access to the Brighten-V1 dataset from the [Brighten Study Public Researcher Portal](https://www.synapse.org/Synapse:syn10848316/wiki/548727) (you must first complete the online trainings and obtain approval to use the data)
2. Various tables must be downloaded manually as .csv files, placed in a directory called "data," and given the names phq9.csv, baseline_phq9.csv, baseline_demographics.csv, auditc.csv, gad7.csv, mania_psychosis.csv, and sds.csv. 
3. Either run ./run_main_analysis.sh (or ./run_all.sh, to include sensitivity analyses) or proceed step by step below for the same result. 
4. Run the following command (having installed Python with the various dependencies): 
```
python clean_mdd_improve_predict_data.py --dest_dir results/main
```
Check that a new file "missing_formatted_Brighten-v1_all.csv" has been created in results/main

5. Run the following command (having installed R with the [miceRanger](https://cran.r-project.org/web/packages/miceRanger/index.html) package):
```
Rscript miceRanger_imputation.r results/main
```
Check that a new file "miceRanger_imputed_formatted_Brighten-v1_all.csv" has been created in results/main

6. To check whether the distributions of imputed variables align with natural values, run the following command: 
```
python plot_imputation_distributions.py --directory results/main
```
This will create new files "binary_vars_distribution.png" and "continuous_vars_distribution.png"


## Forward variable selection and model fitting

All commands for running the forward selection and model fitting are found in forward_selection_commands_main.txt. 

**Please note:** You can run all of these in parallel using the following: 
```
mkdir -p results/main/results_Brighten-v1
./parallel_executor.sh forward_selection_commands_main.txt
```
This will probably take several hours. Results from our execution of this analysis can be found in results/main/results_Brighten-v1 (these will be overwritten if you re-run the analysis).

You can generate a summary table similar to the one in our paper with: 
```
python generate_summary_table.py --directory results/main
```


## Follow-up decision tree analysis and GAD-7/SDS correlation

You can follow the steps of our follow-up decision tree analysis, and exploratory analysis of the correlation between GAD-7 and SDS, in decision_tree_threshold_analysis.ipynb.


## Sensitivity analyses

Please see run_sensitivity_analyses.sh to reproduce our sensitivity analyses. decision_tree_threshold_analysis.ipynb is configurable to run the interpretability analyses for the sensitivity analyses by uncommenting certain directories in early notebook cells. 
