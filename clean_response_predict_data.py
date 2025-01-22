import math
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

def check_duplicates(df, columns=None):
    if columns:
        duplicates = df[df.duplicated(subset=columns, keep=False)]
    else:
        duplicates = df[df.duplicated(keep=False)]
    
    if not duplicates.empty:
        dup_count = len(duplicates)
        dup_rows = duplicates.index.tolist()
        raise ValueError(f"Found {dup_count} duplicate rows. Duplicate row indices: {dup_rows}")

## Load dataset ##

data = pd.read_csv("data/baseline_phq9.csv")
data["base_phq9"] = data.apply(lambda row: sum([row["phq9_" + str(qnum+1)] for qnum in range(9)]), axis=1)
data = data[["participant_id", "study", "base_phq9"]]

check_duplicates(data)

demog = pd.read_csv("data/baseline_demographics.csv")
demog = demog.drop(["study", "heard_about_us"], axis=1)

data = data.merge(demog, on=["participant_id"])

check_duplicates(data)

# Convert start date to datetime format
data["dt_start"] = data.apply(lambda row: datetime.strptime(row["startdate"], '%Y-%m-%d %H:%M:%S'), axis=1)
data = data.drop(["startdate"], axis=1)

check_duplicates(data)

## Format dataset by combining data from multiple files ##

table_names = [
    "gad7", 
    "sds",
    "auditc",
    "mania_psychosis",
]

tables = {key: pd.read_csv("data/" + key + ".csv") for key in table_names}

# Keep only the first chronological entry for each participant
for tn in table_names:
    assert tables[tn].dt_response.isna().sum() == 0
    
    # Get time of the responses in datetime format
    tables[tn]["dt_response"] = tables[tn].apply(
        lambda row: datetime.strptime(row["dt_response"].split("+")[0], '%Y-%m-%d %H:%M:%S'), axis=1)
    
    # Keep only the record with the earliest time for each participant
    tables[tn] = tables[tn].loc[tables[tn].groupby("participant_id").dt_response.idxmin()]
    
    # Get study start time for each participant's entry
    tables[tn] = tables[tn].merge(data[["participant_id", "dt_start"]], on=["participant_id"])
    
    tables[tn]
    
    # Reproduce the "week" field from timestamp data
    tables[tn]["dt_week"] = tables[tn].apply(lambda row: math.ceil((row["dt_response"] - row["dt_start"]).days/7), axis=1)
    
    # Anything with week=0 should be week=1, because week=0 is inconsistently used. 
    tables[tn]["week"] = tables[tn].apply(lambda row: 1 if row["week"] == 0 else row["week"], axis=1)
    tables[tn]["dt_week"] = tables[tn].apply(lambda row: 1 if row["dt_week"] == 0 else row["dt_week"], axis=1)
    
    inc_weeks = tables[tn][tables[tn]["week"] != tables[tn]["dt_week"]]
    print(tn, " number of records with inconsistent weeks: ", len(inc_weeks))
    print(tn, " number of records with week > 2: ", len(tables[tn][tables[tn]["week"] > 2]))
    print(tn, " number of records with week > 3: ", len(tables[tn][tables[tn]["week"] > 3]))
    tables[tn] = tables[tn][tables[tn]["dt_week"] <= 2] # First entry must be within first 2 weeks! 

tables["gad7"] = tables["gad7"][["participant_id", "gad7_sum"]]

tables["sds"]["sds_sum"] = tables["sds"].apply(lambda row: row["sds_1"] + row["sds_2"] + row["sds_3"], axis=1)
tables["sds"] = tables["sds"][["participant_id", "sds_sum"]]

tables["auditc"] = tables["auditc"][["participant_id", "alc_sum"]]

# If any of the first three questions are "1", the patient has some indication of a history of mania. 
tables["mania_psychosis"]["mania_history"] = tables["mania_psychosis"].apply(
    lambda row: 1 if row["screen_1"] + row["screen_2"] + row["screen_3"] > 0 else 0, axis=1)
# The fourth question assesses history of psychosis as a binary variable. 
tables["mania_psychosis"]["psychosis_history"] = tables["mania_psychosis"]["screen_4"]
tables["mania_psychosis"] = tables["mania_psychosis"][["participant_id", "mania_history", "psychosis_history"]]

for tn in table_names:
    data = data.merge(tables[tn], on=["participant_id"], how='left')

check_duplicates(data)

## Getting value of PHQ9 at various weeks into the study ##

phq9 = pd.read_csv("data/phq9.csv")
data_pt_ids = list(data["participant_id"].unique())
phq9 = phq9[phq9["participant_id"].isin(data_pt_ids)]

print("Number of pts with one or more PHQ-9 readings:", len(phq9["participant_id"].unique()))

def get_weekly_average(phq9_data, week):
    week_data = phq9_data[phq9_data["week"] == week]
    week_avg = week_data.groupby("participant_id")["sum_phq9"].mean().reset_index()
    return week_avg

phq9_1week = get_weekly_average(phq9, 1)
print("Number of pts with PHQ-9 at 1 week:", len(phq9_1week))

phq9_2week = get_weekly_average(phq9, 2)
print("Number of pts with PHQ-9 at 2 weeks:", len(phq9_2week))

phq9_3week = get_weekly_average(phq9, 3)
print("Number of pts with PHQ-9 at 3 weeks:", len(phq9_3week))

phq9_4week = get_weekly_average(phq9, 4)
print("Number of pts with PHQ-9 at 4 weeks:", len(phq9_4week))

check_duplicates(data)

for idx, phq9_later in enumerate([phq9_1week, phq9_2week, phq9_3week, phq9_4week]):
  phq9_later[f"week{idx+1}_phq9"] = phq9_later["sum_phq9"]
  phq9_later = phq9_later[["participant_id", f"week{idx+1}_phq9"]]
  data = data.merge(phq9_later, on=["participant_id"], how="left")

check_duplicates(data)

## Filter dataset to keep only participants who started with a clinical level of depression (PHQ9 10 or above at baseline) ##

len_before = len(data)
data = data[data["base_phq9"] >= 10]
print(f"Proportion of pts with base_phq9 of 10 or higher: {len(data)/len_before}")

## Filter dataset to keep only participants who had at least one measure of any kind after the initial intake
# Create a boolean mask for records that have base_phq9 but all other columns are missing
mask = (data['base_phq9'].notna() & 
        data[['gad7_sum', 'sds_sum', 'mania_history', 'psychosis_history', 
              'alc_sum', 'week1_phq9', 'week2_phq9', 'week3_phq9', 'week4_phq9']].isna().all(axis=1))

# Calculate the proportion of participants with only base_phq9/demographics
proportion = mask.mean()
print(f"Proportion of eligible pts with ONLY baseline PHQ-9 and demographics: {proportion:.3f}")
data = data[~mask].reset_index()

print("Total eligible participants:", len(data))

## Convert various variables to suitable formats for machine learning algorithms ##

data['gender'] = data['gender'].replace({'Female': 0})
data['gender'] = data['gender'].apply(lambda x: 1 if x != 0 else x)

data['education'] = data['education'].replace({'None': 0, 'Elementary School': 0, 'High School': 0})
data['education'] = data['education'].apply(lambda x: 1 if x != 0 else x)

data['working'] = data['working'].map({'Yes': 1, 'No': 0})

data['marital_status'] = data['marital_status'].replace({'Married/Partner': 1})
data['marital_status'] = data['marital_status'].apply(lambda x: 0 if x != 1 else x)

data['race'] = data['race'].replace({'Non-Hispanic White': 0})
data['race'] = data['race'].apply(lambda x: 1 if x != 0 else x)

def convert_income_satisfaction(value):
    if value == "Can't make ends meet": return 1
    elif isinstance(value, float) and np.isnan(value): return float('nan')
    else: return 0

data['income_satisfaction'] = data['income_satisfaction'].apply(convert_income_satisfaction)

check_duplicates(data)

# One-hot encode the study_arm column
study_arm_dummies = pd.get_dummies(data['study_arm'], prefix='study_arm', dummy_na=True).astype(int)
data = data.drop('study_arm', axis=1)
data = pd.concat([data, study_arm_dummies], axis=1)

check_duplicates(data)

baseline_features = [
    "age", "gender", "race", "working", "marital_status", "education", "income_satisfaction", 
    "base_phq9", "gad7_sum", "sds_sum", "mania_history", "psychosis_history", "alc_sum",
    "study_arm_EVO", "study_arm_iPST", "study_arm_HealthTips", 
]
outcome_measures = ["week1_phq9", "week2_phq9", "week3_phq9", "week4_phq9"]

data = data[["participant_id", "study"] + baseline_features + outcome_measures]

check_duplicates(data)

for study in ['Brighten-v1']:
    print("-------------------\nSaving data for", study)
    study_data = data[data['study'] == study]

    study_data.to_csv(f"missing_formatted_{study}_all.csv", index=False)

    # Missingness plot
    missing = study_data.isnull()
    matplotlib.use('Agg')
    plt.figure(figsize=(12, 8))
    sns.heatmap(missing, cbar=False, yticklabels=False, cmap='viridis')
    plt.title(f"Missingness Plot for {study}")
    plt.xlabel('Variables')
    plt.ylabel('Observations')
    plt.savefig(f"{study}_missingness.jpg", dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory

    # Create a missingness summary
    missingness = study_data.isnull().sum().sort_values(ascending=False)
    missingness_pct = 100 * missingness / len(study_data)
    missingness_table = pd.concat([missingness, missingness_pct], axis=1, keys=['Missing Values', '% Missing'])
    missingness_table.to_csv(f"{study}_missingness_summary.csv")
    print(f"Missingness summary saved as {study}_missingness_summary.csv")