# Load required libraries
library(miceRanger)
library(dplyr)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Set data directory (default to current directory if not provided)
if (length(args) > 0) {
  data_dir <- args[1]
  # Ensure directory ends with a slash
  if (!endsWith(data_dir, "/")) {
    data_dir <- paste0(data_dir, "/")
  }
} else {
  data_dir <- "./"
}

# Handle second argument for extended PHQ9 weeks
use_extended_weeks <- FALSE
final_week_var <- "week4_phq9"

if (length(args) > 1) {
  if (args[2] == "week6plus_phq9") {
    use_extended_weeks <- TRUE
    final_week_var <- "week12_phq9"
  } else {
    stop(paste("Error: Invalid second argument:", args[2], ". Must be 'week6plus_phq9' or omitted."))
  }
}

# Construct file paths
input_file <- paste0(data_dir, "missing_formatted_Brighten-v1_all.csv")
output_file <- paste0(data_dir, "miceRanger_imputed_formatted_Brighten-v1_all.csv")

# Check if input file exists
if (!file.exists(input_file)) {
  stop(paste("Error: Input file not found:", input_file))
}

# Read the data
data <- read.csv(input_file)

# Define all variables we want to impute
vars_to_impute <- c(
  # Baseline variables
  "base_phq9", "age", "gender",
  "race_is_latino", "race_is_black", "race_is_asian", "race_is_multiracial_or_other", 
  "income_satisfaction", "education", "working", "marital_status",
  "gad7_sum", "sds_sum", "alc_sum", "mania_history",
  "study_arm_HealthTips", "study_arm_EVO", "study_arm_iPST",
  # Outcome variables
  "week1_phq9", "week2_phq9", "week3_phq9", "week4_phq9"
)

# Add extended weeks if requested
if (use_extended_weeks) {
  extended_weeks <- c("week6_phq9", "week8_phq9", "week10_phq9", "week12_phq9")
  vars_to_impute <- c(vars_to_impute, extended_weeks)
}

# Exclude mania history from the list of variables to impute
vars_to_impute <- vars_to_impute[vars_to_impute != "mania_history"]

# Subset data to only variables we want to impute
imputation_data <- data[, vars_to_impute]

# Convert binary variables to factors
binary_vars <- c("gender", "education", "working", "marital_status", "income_satisfaction",
                 "race_is_latino", "race_is_black", "race_is_asian", "race_is_multiracial_or_other",
                 "study_arm_HealthTips", "study_arm_EVO", "study_arm_iPST")

imputation_data[binary_vars] <- lapply(imputation_data[binary_vars], factor)

# Define variables that shouldn't be used as predictors
study_arms <- c("study_arm_HealthTips", "study_arm_EVO", "study_arm_iPST")
phq9_outcomes <- c("week1_phq9", "week2_phq9", "week3_phq9", "week4_phq9")

# Add extended weeks to PHQ9 outcomes if using them
if (use_extended_weeks) {
  extended_weeks <- c("week6_phq9", "week8_phq9", "week10_phq9", "week12_phq9")
  phq9_outcomes <- c(phq9_outcomes, extended_weeks)
}

# Create predictor matrix (TRUE/FALSE instead of 1/0)
pred_matrix <- matrix(TRUE, 
                     nrow = length(vars_to_impute), 
                     ncol = length(vars_to_impute),
                     dimnames = list(vars_to_impute, vars_to_impute))

# Set study arms to not predict any variables except PHQ9 outcomes
for (var in vars_to_impute) {
  if (!var %in% phq9_outcomes) {
    pred_matrix[var, study_arms] <- FALSE
  }
}

# Set up random forest parameters
rf_params <- list(
  num.trees = 100,        # number of trees per forest
  mtry = NULL,           # default sqrt(p) for classification, p/3 for regression
  min.node.size = 5,     # minimum size of terminal nodes
  max.depth = 8       # maximum depth of trees (NULL = unlimited)
)

# Perform imputation
set.seed(123)  # for reproducibility
imp <- miceRanger(
  data = imputation_data,
  m = 100,              # number of imputations
  maxiter = 5,         # number of iterations
  meanMatchCandidates = 5, # Num of candidate donors for PMM
  predictorMatrix = pred_matrix,
  returnModels = FALSE,  # keep models for diagnostics
  verbose = TRUE,       # show progress
  num.trees = rf_params$num.trees,
  min.node.size = rf_params$min.node.size,
  max.depth = rf_params$max.depth,
  valueSelector = "meanMatch"
)

# Convert to long format dataset
final_data <- completeData(imp) %>%
  bind_rows(.id = ".imp") %>%
  mutate(.imp = as.numeric(gsub(".*_", "", tolower(.imp)))) %>%
  rename("_Imputation_" = .imp)

# Add outcome variable
final_data$mdd_improve <- ifelse(final_data[[final_week_var]] < 10 & 
                              final_data[[final_week_var]] <= (final_data$base_phq9 * 0.5), 
                              1, 0)

# Save the imputed data
write.csv(final_data, output_file, row.names = FALSE)