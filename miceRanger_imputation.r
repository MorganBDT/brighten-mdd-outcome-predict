# Load required libraries
library(miceRanger)
library(dplyr)

# Read the data
data <- read.csv("missing_formatted_Brighten-v1_all.csv")

# Define all variables we want to impute
vars_to_impute <- c(
  # Baseline variables
  "base_phq9", "gender", "education", "working",
  "marital_status", "race", "age", "income_satisfaction",
  "gad7_sum", "sds_sum", "alc_sum", "mania_history",
  "psychosis_history", "study_arm_HealthTips",
  "study_arm_EVO", "study_arm_iPST",
  # Outcome variables
  "week1_phq9", "week2_phq9", "week3_phq9", "week4_phq9"
)

# Subset data to only variables we want to impute
imputation_data <- data[, vars_to_impute]

# Convert binary variables to factors
binary_vars <- c("gender", "education", "working", "marital_status",
                 "race", "income_satisfaction", "mania_history",
                 "psychosis_history", "study_arm_HealthTips",
                 "study_arm_EVO", "study_arm_iPST")

imputation_data[binary_vars] <- lapply(imputation_data[binary_vars], factor)

# Define variables that shouldn't be used as predictors
study_arms <- c("study_arm_HealthTips", "study_arm_EVO", "study_arm_iPST")
phq9_outcomes <- c("week1_phq9", "week2_phq9", "week3_phq9", "week4_phq9")

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

# Add response variable
final_data$response <- ifelse(final_data$week4_phq9 < 10 & 
                              final_data$week4_phq9 <= (final_data$base_phq9 * 0.5), 
                              1, 0)

# Save the imputed data
write.csv(final_data, "miceRanger_imputed_formatted_Brighten-v1_all.csv", row.names = FALSE)