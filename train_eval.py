import math
import copy
import pandas as pd
import numpy as np
import random
import ml_models
import functools
import time
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_auc_samples(data, model_class_name, predictor_tuple, outcome, n_bootstrap, seed, model_params_str, resample=False, test_data=None):
    """
    Core function to compute AUC samples for a given model and predictor set.
    
    Args:
        data: pandas DataFrame containing the data
        model_class_name: string name of the model class
        predictor_tuple: tuple of predictor column names
        outcome: string name of the outcome column
        n_bootstrap: number of bootstrap iterations
        seed: random seed
        model_params_str: string representation of model parameters
        resample: whether to resample data with replacement
        test_data: optional pandas DataFrame containing test data. If provided, 
                  all predictions will be made on this test set instead of a 
                  split from the training data.
    
    Returns:
        numpy array of AUC samples
    """
    np.random.seed(seed)
    auc_samples = []
    coefs_samples = [] # For logistic regression only
    printed_warning_train = False
    printed_warning_test = False
    
    for _ in range(n_bootstrap):
        if '_Imputation_' in data.columns:
            imputation = np.random.choice(data['_Imputation_'].unique())
            imputed_data = data[data['_Imputation_'] == imputation]
        else:
            if not printed_warning_train:
                print("WARNING: no imputations in training data")
                printed_warning_train = True
            imputed_data = data
        if resample:
            imputed_data = imputed_data.sample(n=len(imputed_data), replace=True)
        
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(imputed_data, imputed_data[outcome], test_size=0.2)

        if test_data is not None:
            if '_Imputation_' in test_data.columns:
                imputation_test = np.random.choice(test_data['_Imputation_'].unique())
                imputed_test_data = test_data[test_data['_Imputation_'] == imputation_test]
            else:
                if not printed_warning_test:
                    print("WARNING: no imputations in test data")
                    printed_warning_test = True
                imputed_test_data = test_data
            if resample:
                imputed_test_data = imputed_test_data.sample(n=len(imputed_test_data), replace=True)
            _, X_test, _, y_test = train_test_split(imputed_test_data, imputed_test_data[outcome], test_size=0.8)
        
        if predictor_tuple:
            # Initialize StandardScaler
            scaler = StandardScaler()
            
            # Fit scaler on training data and transform both training and test data
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train[list(predictor_tuple)]),
                columns=list(predictor_tuple),
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test[list(predictor_tuple)]),
                columns=list(predictor_tuple),
                index=X_test.index
            )
            
            # Train and predict with scaled data
            model = model_class_map[model_class_name](eval(model_params_str))
            model.fit(X_train_scaled, y_train)
            y_pred = model.proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred)

            if model_class_name == "logistic_regression":
                coefs = pd.Series(np.reshape(model.model.coef_, -1),index=list(predictor_tuple))
            else:
                coefs = None

        else:
            auc = 0.5
        
        auc_samples.append(auc)
        if coefs is not None:
            coefs_samples.append(coefs)

    if len(coefs_samples) > 0:
        coefs_df = pd.DataFrame(coefs_samples)
    else:
        coefs_df = None

    return np.array(auc_samples), coefs_df


# Global counter for cache hits and misses
cache_hits = 0
cache_misses = 0

@functools.lru_cache(maxsize=None)
def _get_cached_auc_samples(model_class_name, predictor_tuple, outcome, n_bootstrap, seed, model_params_str, resample=False):
    global data
    auc_samples, _ = get_auc_samples(data, model_class_name, predictor_tuple, outcome, n_bootstrap, seed, model_params_str, resample)
    return auc_samples


def get_cached_auc_samples(model_class_name, predictor_tuple, outcome, n_bootstrap, seed, model_params_str, verbose=False, resample=False):
    """
    This function computes and caches AUC samples for a given model and predictor set.
    It also provides debug output for cache hits and misses.
    """
    global cache_hits, cache_misses
    
    start_time = time.time()
    cache_info_before = _get_cached_auc_samples.cache_info()
    
    result = _get_cached_auc_samples(model_class_name, predictor_tuple, outcome, n_bootstrap, seed, model_params_str, resample=resample)
    
    cache_info_after = _get_cached_auc_samples.cache_info()
    end_time = time.time()
    duration = end_time - start_time
    
    if cache_info_after.hits > cache_info_before.hits:
        cache_hits += 1
        if verbose:
            print(f"Cache hit for {model_class_name} with {len(predictor_tuple)} predictors. Time: {duration:.4f} seconds")
    else:
        cache_misses += 1
        if verbose:
            print(f"Cache miss for {model_class_name} with {len(predictor_tuple)} predictors. Time: {duration:.4f} seconds")
    
    if verbose:
        print(f"Total cache hits: {cache_hits}, Total cache misses: {cache_misses}")
    
    return result

def bootstrap_auc_comparison(data, predictor_list, new_predictor, outcome, model_class, ml_model_args, n_bootstrap, seed, resample=False):
    model_class_name = model_class.model_name()
    predictor_tuple = tuple(sorted(predictor_list))  # Convert list to sorted tuple for hashing
    model_params_str = str(ml_model_args)  # Convert model parameters to string for hashing
    
    if not predictor_list:
        # Base case: compare new predictor against AUC=0.5
        auc_k_samples = np.full(n_bootstrap, 0.5)
        new_predictor_tuple = tuple([new_predictor])
    else:
        # Get cached AUC samples for the current model
        auc_k_samples = get_cached_auc_samples(model_class_name, predictor_tuple, outcome, n_bootstrap, seed, model_params_str, resample=resample)
        # Prepare tuple for the model with the new predictor
        new_predictor_tuple = tuple(sorted(predictor_list + [new_predictor]))
    
    # Get AUC samples for the model with the new predictor
    if model_class_name == "logistic_regression":
        auc_new_samples, coefs_df = get_auc_samples(data, model_class_name, new_predictor_tuple, outcome, n_bootstrap, seed, model_params_str, resample=resample)

        # Calculate statistics for each coefficient
        mean_coefs = {}
        coefs_ci_lower = {}
        coefs_ci_upper = {}
        
        for feature in coefs_df.columns:
            mean_coefs[feature] = np.mean(coefs_df[feature])
            coefs_ci_lower[feature] = np.percentile(coefs_df[feature], 2.5)
            coefs_ci_upper[feature] = np.percentile(coefs_df[feature], 97.5)
    else:
        auc_new_samples = get_cached_auc_samples(model_class_name, new_predictor_tuple, outcome, n_bootstrap, seed, model_params_str, resample=resample)
        mean_coefs, coefs_ci_lower, coefs_ci_upper = None, None, None

    mean_auc = np.mean(auc_new_samples)
    
    # Calculate AUC differences
    auc_diff = auc_new_samples - auc_k_samples
    
    mean_diff = np.mean(auc_diff)
    
    ci_lower = np.percentile(auc_diff, 2.5)
    ci_upper = np.percentile(auc_diff, 97.5)
    p_value = np.mean(auc_diff <= 0)  # One-sided p-value
    
    return mean_auc, mean_diff, ci_lower, ci_upper, p_value, mean_coefs, coefs_ci_lower, coefs_ci_upper

# Global variable to map model class names to actual classes
model_class_map = {
    'logistic_regression': ml_models.LogisticRegression,
    'decision_tree': ml_models.DecisionTree,
    'random_forest': ml_models.RandomForest,
    'svm': ml_models.SVM,
    'knn': ml_models.KNN,
}

# Global variable for data
data = None

def set_global_data(global_data):
    global data
    data = global_data