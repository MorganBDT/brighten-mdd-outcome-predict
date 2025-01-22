import sys
import numpy as np
import pandas as pd
import argparse
import time
import warnings
from collections import deque
import concurrent.futures
from functools import partial
import load_data
import ml_models
import train_eval

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

def get_args(argv):
    parser = argparse.ArgumentParser(description="Forward selection with bootstrap confidence intervals")
    
    parser.add_argument('--filename', type=str, default="formatted_brighten.csv", help="Path to input csv file")
    parser.add_argument('--output_filename', type=str, default=None, help="Path to output csv file")
    parser.add_argument('--svm', default=False, action='store_true')
    parser.add_argument('--logistic_regression', default=False, action='store_true')
    parser.add_argument('--random_forest', default=False, action='store_true')
    parser.add_argument('--decision_tree', default=False, action='store_true')
    parser.add_argument('--knn', default=False, action='store_true')
    parser.add_argument('--standardize_on_load', default=False, action='store_true', help="If selected, standardize the data when initially loading it. Not recommended - we standardize based on the training data inside get_auc_samples (train_eval.py)")
    parser.add_argument('--stratified_kfold', default=False, action='store_true', help="Use stratified k-fold CV")
    parser.add_argument('--xval_folds', type=int, default=5, help="Number of folds in k-fold cross-validation")
    parser.add_argument('--multiple_imputation', default=False, action='store_true', help="Use multiply-imputed dataset")
    parser.add_argument('--n_imputations', type=int, default=100, help="Number of imputations in multiple imputation setup")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--metric', type=str, default="auc", help="Metric to use for variable selection. 'auc' or 'accuracy'")
    parser.add_argument('--shuffle_outcomes', default=False, action='store_true', help="Shuffle the outcomes (sanity check)")
    parser.add_argument('--resample', default=False, action='store_true', help="Resample the dataset during bootstrap procedure")
    parser.add_argument('--metric_improvement_margin', type=float, default=0.02, help="Minimum AUC improvement to continue selection")
    parser.add_argument('--bh_correction', default=False, action='store_true', help="Apply Benjamini-Hochberg correction")
    parser.add_argument('--beam_width', type=int, default=None, help="Maximum number of paths to explore at each step. If None, explores all possible paths.")
    parser.add_argument('--n_bootstrap', type=int, default=2500, help="Number of bootstrap samples")
    parser.add_argument('--max_workers', type=int, default=None, help="Maximum number of worker processes. If None, uses the number of CPU cores.")
    parser.add_argument('--complexity', type=int, default=None, help="Set this to run at a specific complexity (i.e., max_tree_depth for decision tree and random forest)")

    return parser.parse_args(argv)

def evaluate_predictor(data, current_predictors, pred, outcome, model_class, model_params, n_bootstrap, seed, resample):
    try:
        mean_auc, mean_diff, ci_lower, ci_upper, p_value, mean_coefs, coefs_ci_lower, coefs_ci_upper = train_eval.bootstrap_auc_comparison(
            data, current_predictors, pred, outcome, model_class, model_params, n_bootstrap, seed, resample=resample
        )
        return {
            "predictor": pred,
            "mean_auc": mean_auc,
            "auc_increase": mean_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": p_value,
            "mean_coefs": mean_coefs,
            "coefs_ci_lower": coefs_ci_lower, 
            "coefs_ci_upper": coefs_ci_upper,
        }
    except Exception as e:
        print(f"Error evaluating predictor {pred}: {str(e)}")
        return None
    
def benjamini_hochberg_correction(p_values, fdr=0.05):
    """
    Perform Benjamini-Hochberg correction on a list of p-values.
    Returns a list of booleans indicating which p-values are significant.
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]
    
    thresholds = np.arange(1, n + 1) * fdr / n
    last_significant = np.where(sorted_p_values < thresholds)[0]
    
    if len(last_significant) == 0:
        return np.zeros(n, dtype=bool)
    
    last_significant = last_significant[-1]
    significant = np.zeros(n, dtype=bool)
    significant[sorted_indices[:last_significant + 1]] = True
    
    return significant

def explore_paths(data, predictors, outcome, model_class, model_params, args):
    paths = deque([{"predictors": [], "auc": 0.5}])
    final_models = []
    explored_predictor_sets = set()  # Set to keep track of explored predictor combinations
    
    while paths:
        current_path = paths.popleft()
        current_predictors = current_path["predictors"]
        current_auc = current_path["auc"]
        
        # Convert current predictors to a frozenset for hashing
        current_predictor_set = frozenset(current_predictors)
        
        # Skip this path if we've already explored this combination of predictors
        if current_predictor_set in explored_predictor_sets:
            continue
        
        # Mark this combination as explored
        explored_predictor_sets.add(current_predictor_set)
        
        print(f"\nExploring path with predictors: {current_predictors}")
        print(f"Current AUC: {current_auc:.4f}")
        
        unused_predictors = [p for p in predictors if p not in current_predictors]

        if args.resample:
            raise NotImplementedError("Resampling is broken - it should only be resampling the training set on each iteration, not the whole dataset (otherwise there is leakage between train and validation sets, especially for KNN)")
        
        # Parallel execution using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            eval_func = partial(
                evaluate_predictor, 
                data, 
                current_predictors, 
                outcome=outcome, 
                model_class=model_class, 
                model_params=model_params, 
                n_bootstrap=args.n_bootstrap, 
                seed=args.seed,
                resample=args.resample
            )
            results = list(executor.map(eval_func, unused_predictors))
        
        # Filter out None results (failed evaluations) and print results
        results = [r for r in results if r is not None]
        for result in results:
            print_str = f"Predictor: {result['predictor']}, AUC increase: {result['auc_increase']:.4f}, 95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}], p-value: {result['p_value']:.4f}"
            if model_class.model_name() == "logistic_regression":
                print_str = print_str + f", Mean coefs: {result['mean_coefs']}, Coefs CI lower: {result['coefs_ci_lower']}, Coefs CI upper: {result['coefs_ci_upper']}"
            print(print_str)
        
        if not results:
            print("No valid predictors found. Stopping exploration for this path.")
            if current_predictors:
                final_models.append(current_path)
            continue
        
        results_df = pd.DataFrame(results)

        if args.bh_correction: # Apply Benjamini-Hochberg correction
            significant = benjamini_hochberg_correction(results_df['p_value'])
            significant_predictors = results_df[
                significant & 
                (results_df['auc_increase'] > args.metric_improvement_margin)
            ].sort_values('auc_increase', ascending=False)
        else: # Use a global p-value threshold (less conservative, higher FDR)
            significant_predictors = results_df[
                (results_df['p_value'] < 0.05) & 
                (results_df['auc_increase'] > args.metric_improvement_margin)
            ].sort_values('auc_increase', ascending=False)
        
        if significant_predictors.empty:
            print("No predictors significantly improved the model. Stopping exploration for this path.")
            if current_predictors:  # Only add to final models if we have at least one predictor
                final_models.append(current_path)
            continue
        
        # Apply beam width limitation if specified
        if args.beam_width is not None:
            significant_predictors = significant_predictors.head(args.beam_width)
            print(f"Beam search: keeping top {args.beam_width} predictors")
        
        for _, predictor in significant_predictors.iterrows():
            new_predictors = current_predictors + [predictor['predictor']]
            new_predictor_set = frozenset(new_predictors)
            
            # Only add this new path if we haven't explored this combination before
            if new_predictor_set not in explored_predictor_sets:
                new_auc = current_auc + predictor['auc_increase']
                
                print(f"Adding predictor {predictor['predictor']}. New AUC: {new_auc:.4f}")
                
                new_path = {
                    "predictors": new_predictors,
                    "auc": new_auc,
                    "ci_lower": predictor['ci_lower'],
                    "ci_upper": predictor['ci_upper'],
                    "p_value": predictor['p_value']
                }
                
                paths.append(new_path)
            else:
                print(f"NOT adding predictor {predictor['predictor']}, because this combination of predictors is already explored. New AUC: {new_auc:.4f}")

    return final_models

def main():
    start_time = time.time()
    args = get_args(sys.argv[1:])
    
    model_classes = [model_class for model_class in [
        ml_models.LogisticRegression, 
        ml_models.DecisionTree,
        ml_models.KNN,
        ml_models.SVM, 
        ml_models.RandomForest,
    ] if args.__dict__[model_class.model_name()]]
    
    if len(model_classes) == 0:
        print("You must specify one or more model types. Exiting...")
        sys.exit(0)
        
    outcomes = ["response"]
    
    data, predictors = load_data.load_data(standardize=args.standardize_on_load, filename=args.filename, impute=False)

    train_eval.set_global_data(data)

    all_results = []
    for model_class in model_classes:
        for outcome in outcomes:
            print(f"\n------- {model_class.model_name()} predicting {outcome} -------")

            if args.complexity is not None:
                complexity_range = range(args.complexity, args.complexity+1)
            elif model_class.model_name() == "decision_tree":
                complexity_range = range(1, 7)
            elif model_class.model_name() == "random_forest":
                complexity_range = range(1, 7)
            else:
                complexity_range = range(1)

            for complexity in complexity_range:
                rf_n_estimators = 100
                model_params = {
                    "dt_max_depth": complexity,
                    "rf_n_estimators": rf_n_estimators,
                    "rf_max_depth": complexity,
                }

                print(f"\nExploring models with complexity: {complexity}")
                
                final_models = explore_paths(data, predictors, outcome, model_class, model_params, args)
                
                for model in final_models:
                    all_results.append({
                        "outcome": outcome,
                        "model": model_class.model_name(),
                        "predictors": model["predictors"],
                        "predictor_added": model["predictors"][-1] if model["predictors"] else None,
                        "auc": model["auc"],
                        "ci_lower": model.get("ci_lower"),
                        "ci_upper": model.get("ci_upper"),
                        "p_value": model.get("p_value"),
                        "complexity": complexity,
                        "rf_n_estimators": rf_n_estimators
                    })

    all_results_df = pd.DataFrame(all_results)
    print(all_results_df)
    
    if args.output_filename is not None:
        fname = args.output_filename
    else:
        fname = f"forward_selection_results_bootstrap_{args.seed}_{int(time.time())}.csv"
    print(f"SAVING AS: {fname}")
    all_results_df.to_csv(fname, index=False)
    print(f"Total time (seconds): {time.time() - start_time}")

if __name__ == '__main__':
    main()