import numpy as np
import math
import scipy.stats as stats
from statistics import mean


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


def rubin_combine(vals, val_vars, log_normal=False):
    """Combines values from multiple imputations using Rubin's rules.
    
    Args:
        vals: list of values to combine.
        val_vars: list of within-imputation variances. Note that these should already be in log form if log_normal=True.
        log_normal: Bool, set to True for log-normally distributed values (e.g. odds ratios)
                   and False for normally distributed values
        return_p_val: Bool, if True, the fourth return value is the p-value. If False, it's the original list of values.
    """
            
    # Pool the values according to Rubin's rules
    if log_normal:
        transformed_vals = [math.log(val) for val in vals]
        mean_val = mean(transformed_vals)
        pooled_val = math.exp(mean_val)
    else:
        mean_val = mean(vals)
        pooled_val = mean_val
    
    # Mean within-imputation variance
    mean_imp_var = mean(val_vars)
    
    # Between-imputation variance
    n_impute = len(vals)
    if log_normal:
        between_imp_var = np.sum((np.array(transformed_vals) - mean_val) ** 2) / (n_impute - 1)
    else:
        between_imp_var = np.sum((np.array(vals) - mean_val) ** 2) / (n_impute - 1)
                                
    # Total variance, combining within and between imputations
    total_var = mean_imp_var + (1 + (1/n_impute))*between_imp_var
                                
    # Degrees of freedom (https://bookdown.org/mwheymans/bookmi/rubins-rules.html)
    if mean_imp_var <= 0:
        print("Warning: Mean within-imputation variance is non-positive. This typically occurs for variables that are redundant due to multicollinearity.")
        assert mean_imp_var == 0, "Mean imputation variance is less than 0, something is very wrong indeed"
        r = float('inf')
    else:
        r = ((1 + 1/n_impute) * between_imp_var) / mean_imp_var

    if r == float('inf'):
        nu = float('inf')
    else:
        nu = (n_impute - 1) * (1 + 1/r)**2
    
    alpha = 0.05
    p_value = np.nan
    
    # Handle cases where total_var is non-positive or nu is not valid
    if total_var > 0 and not np.isnan(nu):
        t_stat = mean_val / np.sqrt(total_var)
        if nu == float('inf'): # Use normal distribution for infinite DoF
            t_val = stats.norm.ppf(1 - alpha/2)
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        else: # Use t-distribution
            t_val = stats.t.ppf(1 - alpha/2, df=nu)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=nu))

        if log_normal:
            lower_ci = math.exp(mean_val - t_val * np.sqrt(total_var))
            upper_ci = math.exp(mean_val + t_val * np.sqrt(total_var))
        else:
            lower_ci = mean_val - t_val * np.sqrt(total_var)
            upper_ci = mean_val + t_val * np.sqrt(total_var)
    else:
        if total_var <= 0:
            print("Warning: Total variance is non-positive. This means the coefficient and its standard error were constant (likely zero) across all imputations.")
        if np.isnan(nu):
            print("Warning: Degrees of freedom is NaN. This occurs when both within- and between-imputation variance are zero, making the test statistic undefined.")
        lower_ci, upper_ci = np.nan, np.nan

    return pooled_val, lower_ci, upper_ci, vals, p_value