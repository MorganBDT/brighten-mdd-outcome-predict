import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot distributions of original vs imputed data')
parser.add_argument('--directory', '-d', type=str, default='.', 
                    help='Directory containing the CSV files (default: current directory)')
parser.add_argument('--week6plus_phq9', action='store_true',
                    help='Include PHQ9 measurements for weeks 6, 8, 10, and 12 in the analysis')
args = parser.parse_args()

# Construct file paths
original_file = os.path.join(args.directory, 'missing_formatted_Brighten-v1_all.csv')
imputed_file = os.path.join(args.directory, 'miceRanger_imputed_formatted_Brighten-v1_all.csv')

# Construct output file paths
continuous_plot_file = os.path.join(args.directory, 'continuous_vars_distribution.png')
binary_plot_file = os.path.join(args.directory, 'binary_vars_distribution.png')

# Read the data
original_df = pd.read_csv(original_file)
imputed_df = pd.read_csv(imputed_file)

# Replace any infinite values with NaN
original_df = original_df.replace([np.inf, -np.inf], np.nan)
imputed_df = imputed_df.replace([np.inf, -np.inf], np.nan)

# List of features to analyze - conditionally include week 6+ PHQ9 measures
base_continuous_vars = ['gad7_sum', 'sds_sum', 'alc_sum', 'week1_phq9', 'week2_phq9', 'week3_phq9', 'week4_phq9']
if args.week6plus_phq9:
    continuous_vars = base_continuous_vars + ['week6_phq9', 'week8_phq9', 'week10_phq9', 'week12_phq9']
    print("Including PHQ9 measurements for weeks 6, 8, 10, and 12")
else:
    continuous_vars = base_continuous_vars
    print("Excluding PHQ9 measurements for weeks 6, 8, 10, and 12")

binary_vars = ['income_satisfaction', 'race_is_latino', 'race_is_black', 'race_is_asian', 'race_is_multiracial_or_other', 'gender', 'education', 'working', 'marital_status']

# Set the style for better-looking plots
plt.style.use('default')
sns.set_theme(style="whitegrid")

# Create a figure for continuous variables
fig, axes = plt.subplots(1, len(continuous_vars), figsize=(5*len(continuous_vars), 5))

# Plot continuous variables
for idx, (var, ax) in enumerate(zip(continuous_vars, axes)):
    # Get original non-missing values
    original_values = original_df[var].dropna()
    # Clean any remaining infinite values
    original_values = original_values.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Get imputed values (only for rows where original was missing)
    missing_indices = original_df[original_df[var].isna()].index
    imputed_values = []
    
    # Collect imputed values for all missing indices across all imputations
    for imp_num in range(1, 101):
        imp_data = imputed_df[imputed_df['_Imputation_'] == imp_num]
        imp_subset = imp_data.loc[imp_data.index.isin(missing_indices), var]
        imputed_values.extend(imp_subset.tolist())
    
    imputed_values = pd.Series(imputed_values).dropna()
    # Clean any remaining infinite values
    imputed_values = imputed_values.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Create density plots using matplotlib instead of seaborn
    if len(original_values) > 0:
        original_kde = sns.kdeplot(data=original_values, ax=ax, label='Original', 
                                 color='blue', alpha=0.6, warn_singular=False)
    if len(imputed_values) > 0:
        imputed_kde = sns.kdeplot(data=imputed_values, ax=ax, label='Imputed', 
                                color='red', alpha=0.6, warn_singular=False)
    
    ax.set_title(f'Distribution of {var}')
    ax.set_xlabel(var)
    ax.set_ylabel('Density')
    ax.legend()

plt.tight_layout()
plt.savefig(continuous_plot_file, dpi=300, bbox_inches='tight')
plt.close()

# Plot binary variables
fig, axes = plt.subplots(1, len(binary_vars), figsize=(5*len(binary_vars), 5))
for idx, (var, ax) in enumerate(zip(binary_vars, axes)):
    # Calculate proportions for original non-missing values
    original_props = original_df[var].value_counts(normalize=True)
    
    # Get imputed values for missing entries
    missing_indices = original_df[original_df[var].isna()].index
    imputed_values = []
    
    # Collect imputed values for all missing indices across all imputations
    for imp_num in range(1, 101):
        imp_data = imputed_df[imputed_df['_Imputation_'] == imp_num]
        imp_subset = imp_data.loc[imp_data.index.isin(missing_indices), var]
        imputed_values.extend(imp_subset.tolist())
    
    # Convert imputed values to float type to match original data
    imputed_values = [float(x) for x in imputed_values]
    imputed_values = pd.Series(imputed_values).dropna()
    # Clean any remaining infinite values
    imputed_values = imputed_values.replace([np.inf, -np.inf], np.nan).dropna()

    # Get unique values from both original and imputed data
    all_values = sorted(set([0.0, 1.0]))  # Explicitly set as floats
    
    # Initialize series with all possible values and float dtype
    imputed_props = pd.Series(0.0, index=all_values, dtype=float)
    original_props = original_props.reindex(all_values, fill_value=0.0)
    
    # Update with actual proportions
    value_counts = imputed_values.value_counts(normalize=True)
    imputed_props.update(value_counts)
    
    # Prepare data for plotting
    x = np.arange(len(all_values))
    width = 0.35
    
    # Create bars with aligned indices
    ax.bar(x - width/2, original_props, width, label='Original', color='blue', alpha=0.6)
    ax.bar(x + width/2, imputed_props, width, label='Imputed', color='red', alpha=0.6)
    
    ax.set_title(f'Distribution of {var}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Proportion')
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(val)) for val in all_values])  # Convert to int for display
    ax.legend()

plt.tight_layout()
plt.savefig(binary_plot_file, dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print("-" * 50)

for var in continuous_vars:
    print(f"\n{var}:")
    print("Original (non-missing):")
    print(original_df[var].describe())
    print("\nImputed (previously missing):")
    
    missing_indices = original_df[original_df[var].isna()].index
    imputed_values = []
    for imp_num in range(1, 101):
        imp_data = imputed_df[imputed_df['_Imputation_'] == imp_num]
        imp_subset = imp_data.loc[imp_data.index.isin(missing_indices), var]
        imputed_values.extend(imp_subset.dropna().tolist())
    
    imputed_values = pd.Series(imputed_values)
    # Clean any remaining infinite values
    imputed_values = imputed_values.replace([np.inf, -np.inf], np.nan).dropna()
    print(imputed_values.describe())

for var in binary_vars:
    print(f"\n{var}:")
    print("Original proportions:")
    print(original_df[var].value_counts(normalize=True))
    print("\nImputed proportions:")
    
    missing_indices = original_df[original_df[var].isna()].index
    imputed_values = []
    for imp_num in range(1, 101):
        imp_data = imputed_df[imputed_df['_Imputation_'] == imp_num]
        imp_subset = imp_data.loc[imp_data.index.isin(missing_indices), var]
        imputed_values.extend(imp_subset.dropna().tolist())
    
    imputed_values = pd.Series(imputed_values)
    # Clean any remaining infinite values
    imputed_values = imputed_values.replace([np.inf, -np.inf], np.nan).dropna()
    print(imputed_values.value_counts(normalize=True))