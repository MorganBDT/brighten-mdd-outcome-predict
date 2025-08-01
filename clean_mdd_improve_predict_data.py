import math
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import argparse
import os

def check_duplicates(df, columns=None):
    if columns:
        duplicates = df[df.duplicated(subset=columns, keep=False)]
    else:
        duplicates = df[df.duplicated(keep=False)]
    
    if not duplicates.empty:
        dup_count = len(duplicates)
        dup_rows = duplicates.index.tolist()
        raise ValueError(f"Found {dup_count} duplicate rows. Duplicate row indices: {dup_rows}")

def parse_baseline_cutoff(cutoff_str):
    """Parse baseline cutoff argument and return (cutoff_value, cutoff_type)"""
    if cutoff_str.endswith('_weeks'):
        weeks = int(cutoff_str.split('_')[0])
        if weeks not in [1, 2, 3, 4]:
            raise ValueError("Weeks must be 1, 2, 3, or 4")
        return weeks, 'weeks'
    elif cutoff_str.endswith('_days'):
        days = int(cutoff_str.split('_')[0])
        if not (0 <= days <= 28):
            raise ValueError("Days must be between 0 and 28")
        return days, 'days'
    elif cutoff_str == '2_weeks':
        return 2, 'weeks'
    else:
        raise ValueError("Invalid baseline_cutoff format. Use 'n_weeks' (n=1-4) or 'n_days' (n=0-28)")

def main():
    parser = argparse.ArgumentParser(description='Clean response prediction data with configurable parameters')
    parser.add_argument('--week6plus_phq9', action='store_true', 
                        help='Include PHQ9 measurements for weeks 6, 8, 10, and 12 in the analysis')
    parser.add_argument('--baseline_cutoff', default='5_days', type=str,
                        help='Baseline cutoff for first entry. Format: "n_weeks" (n=1-4) or "n_days" (n=0-28). Default: "5_days"')
    parser.add_argument('--exclude_minimal_baseline_pts', action='store_true', 
                        help='Exclude participants who have only baseline PHQ9 and demographics. If not set, these participants are included.')
    parser.add_argument('--dest_dir', default='.', type=str,
                        help='Directory to save output files. Default: current directory')
    parser.add_argument('--studies_to_include', nargs='+', default=['Brighten-v1'],
                        help='Studies to include in the analysis. Default: Brighten-v1 only. Can also do --studies_to_include Brighten-v1 Brighten-v2')
    
    args = parser.parse_args()
    
    # Parse the baseline cutoff
    cutoff_value, cutoff_type = parse_baseline_cutoff(args.baseline_cutoff)
    print(f"Using baseline cutoff: {cutoff_value} {cutoff_type}")
    
    if args.week6plus_phq9:
        print("Including PHQ9 measurements for weeks 6, 8, 10, and 12")
    else:
        print("Excluding PHQ9 measurements for weeks 6, 8, 10, and 12")
    
    # Create destination directory if it doesn't exist
    os.makedirs(args.dest_dir, exist_ok=True)
    print(f"Output files will be saved to: {os.path.abspath(args.dest_dir)}")

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

        # Create a "dt_day" field based on timestamp data. Day 0 is the day of enrollment (dt_start)
        tables[tn]["dt_day"] = tables[tn].apply(lambda row: (row["dt_response"] - row["dt_start"]).days, axis=1)
        
        # Reproduce the "week" field from timestamp data
        tables[tn]["dt_week"] = tables[tn].apply(lambda row: math.ceil((row["dt_response"] - row["dt_start"]).days/7), axis=1)
        
        # Anything with week=0 should be week=1, because week=0 is inconsistently used (by default, week is "0" if it is less than 1 day after enrollment, or 1 if between 1 and 7 days)
        tables[tn]["week"] = tables[tn].apply(lambda row: 1 if row["week"] == 0 else row["week"], axis=1)
        tables[tn]["dt_week"] = tables[tn].apply(lambda row: 1 if row["dt_week"] == 0 else row["dt_week"], axis=1)

        
        inc_weeks = tables[tn][tables[tn]["week"] != tables[tn]["dt_week"]]
        print(tn, "total number of records: ", len(tables[tn]))
        print(tn, " number of records with inconsistent weeks: ", len(inc_weeks))
        print(tn, " number of records with dt_day >= 3: ", len(tables[tn][tables[tn]["dt_day"] >= 3]))
        print(tn, " number of records with dt_day >= 5: ", len(tables[tn][tables[tn]["dt_day"] >= 5]))
        print(tn, " number of records with week > 1: ", len(tables[tn][tables[tn]["week"] > 1]))
        print(tn, " number of records with week > 2: ", len(tables[tn][tables[tn]["week"] > 2]))
        print(tn, " number of records with week > 3: ", len(tables[tn][tables[tn]["week"] > 3]))
        
        # Apply baseline cutoff based on command-line argument
        if tn not in ["mania_psychosis"]:
            if cutoff_type == 'weeks':
                tables[tn] = tables[tn][tables[tn]["dt_week"] <= cutoff_value]
                print(f"{tn} filtered to entries within {cutoff_value} weeks")
            else:  # cutoff_type == 'days'
                tables[tn] = tables[tn][tables[tn]["dt_day"] <= cutoff_value]
                print(f"{tn} filtered to entries within {cutoff_value} days")

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

    phq9_6week = get_weekly_average(phq9, 6)
    print("Number of pts with PHQ-9 at 6 weeks:", len(phq9_6week))

    phq9_8week = get_weekly_average(phq9, 8)
    print("Number of pts with PHQ-9 at 8 weeks:", len(phq9_8week))

    phq9_10week = get_weekly_average(phq9, 10)
    print("Number of pts with PHQ-9 at 10 weeks:", len(phq9_10week))

    phq9_12week = get_weekly_average(phq9, 12)
    print("Number of pts with PHQ-9 at 12 weeks:", len(phq9_12week))

    check_duplicates(data)

    # Conditionally include week 6+ PHQ9 measures based on command-line argument
    if args.week6plus_phq9:
        week_numbers = [1, 2, 3, 4, 6, 8, 10, 12]
        phq9_weeks = [phq9_1week, phq9_2week, phq9_3week, phq9_4week, phq9_6week, phq9_8week, phq9_10week, phq9_12week]
    else:
        week_numbers = [1, 2, 3, 4]
        phq9_weeks = [phq9_1week, phq9_2week, phq9_3week, phq9_4week]

    for week_num, phq9_later in zip(week_numbers, phq9_weeks):
      phq9_later[f"week{week_num}_phq9"] = phq9_later["sum_phq9"]
      phq9_later = phq9_later[["participant_id", f"week{week_num}_phq9"]]
      data = data.merge(phq9_later, on=["participant_id"], how="left")

    check_duplicates(data)

    ## Convert various variables to suitable formats for machine learning algorithms ##

    data['gender'] = data['gender'].replace({'Female': 0})
    data['gender'] = data['gender'].apply(lambda x: 1 if x != 0 else x)

    data['education'] = data['education'].replace({'None': 0, 'Elementary School': 0, 'High School': 0})
    data['education'] = data['education'].apply(lambda x: 1 if x != 0 else x)

    data['working'] = data['working'].map({'Yes': 1, 'No': 0})

    data['marital_status'] = data['marital_status'].replace({'Married/Partner': 1})
    data['marital_status'] = data['marital_status'].apply(lambda x: 0 if x != 1 else x)

    # Store original race NaNs to propagate them to the one-hot encoded columns
    is_race_nan = data['race'].isna()

    # Create one-hot encoded features for race/ethnicity.
    # 'Non-Hispanic White' will be the reference category (all new columns will be 0).
    # We use float type to accommodate NaNs.
    data['race_is_latino'] = (data['race'] == 'Hispanic/Latino').astype(float)
    data['race_is_black'] = (data['race'] == 'African-American/Black').astype(float)
    data['race_is_asian'] = (data['race'] == 'Asian').astype(float)
    data['race_is_multiracial_or_other'] = data['race'].isin(['More than one', 'Native Hawaiian/other Pacific Islander', 'American Indian/Alaskan Native', 'Other']).astype(float)

    race_cols = ['race_is_latino', 'race_is_black', 'race_is_asian', 'race_is_multiracial_or_other']
    
    # Where original race was NaN, set all new race columns to NaN for later imputation.
    data.loc[is_race_nan, race_cols] = np.nan

    print("Number of nans in race/ethnicity column:", data['race'].isna().sum())

    # Now that we have the one-hot encoded columns, we can drop the original 'race' column.
    data = data.drop('race', axis=1)

    # Old version: binarized "Non-Hispanic White" 0 vs all other 1
    # data['race'] = data['race'].replace({'Non-Hispanic White': 0})
    # data['race'] = data['race'].apply(lambda x: 1 if x != 0 else x)

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
        "age", "gender", "working", "marital_status", "education", "income_satisfaction",
        "race_is_latino", "race_is_black", "race_is_asian", "race_is_multiracial_or_other",
        "base_phq9", "gad7_sum", "sds_sum", "mania_history", "alc_sum",
        "study_arm_EVO", "study_arm_iPST", "study_arm_HealthTips", 
    ]

    # Conditionally define outcome measures based on command-line argument
    if args.week6plus_phq9:
        outcome_measures = ["week1_phq9", "week2_phq9", "week3_phq9", "week4_phq9", "week6_phq9", "week8_phq9", "week10_phq9", "week12_phq9"]
    else:
        outcome_measures = ["week1_phq9", "week2_phq9", "week3_phq9", "week4_phq9"]

    data = data[["participant_id", "study"] + baseline_features + outcome_measures]

    check_duplicates(data)

    # Save unfiltered dataset
    csv_path = os.path.join(args.dest_dir, f"formatted_unfiltered_all.csv")
    data.to_csv(csv_path, index=False)
    for study in args.studies_to_include:
        study_data = data[data['study'] == study]
        csv_path = os.path.join(args.dest_dir, f"formatted_unfiltered_{study}.csv")
        study_data.to_csv(csv_path, index=False)

    ## Filter dataset to keep only participants who started with a clinical level of depression (PHQ9 10 or above at baseline) ##

    len_before = len(data)
    len_before_brightenv1 = len(data[data["study"] == "Brighten-v1"])
    len_before_brightenv2 = len(data[data["study"] == "Brighten-v2"])
    data = data[data["base_phq9"] >= 10].reset_index(drop=True)
    print(f"Proportion of pts with base_phq9 of 10 or higher: {len(data)}/{len_before} = {len(data)/len_before:.3f}")
    print(f"Proportion of pts with base_phq9 of 10 or higher in Brighten-v1: {len(data[data['study'] == 'Brighten-v1'])}/{len_before_brightenv1} = {len(data[data['study'] == 'Brighten-v1'])/len_before_brightenv1:.3f}")
    print(f"Proportion of pts with base_phq9 of 10 or higher in Brighten-v2: {len(data[data['study'] == 'Brighten-v2'])}/{len_before_brightenv2} = {len(data[data['study'] == 'Brighten-v2'])/len_before_brightenv2:.3f}")

    if args.exclude_minimal_baseline_pts:
        ## Filter dataset to keep only participants who had at least one measure of any kind after the initial intake
        # Create a boolean mask for records that have base_phq9 but all other columns are missing
        if args.week6plus_phq9:
            outcome_columns = ['gad7_sum', 'sds_sum', 'mania_history', 
                            'alc_sum', 'week1_phq9', 'week2_phq9', 'week3_phq9', 'week4_phq9',
                            'week6_phq9', 'week8_phq9', 'week10_phq9', 'week12_phq9']
        else:
            outcome_columns = ['gad7_sum', 'sds_sum', 'mania_history', 
                            'alc_sum', 'week1_phq9', 'week2_phq9', 'week3_phq9', 'week4_phq9']

        mask = (data['base_phq9'].notna() & 
                data[outcome_columns].isna().all(axis=1))

        # Calculate the proportion of participants with only base_phq9/demographics
        print(f"Proportion of eligible pts with ONLY baseline PHQ-9 and demographics: {mask.sum()}/{len(mask)} = {mask.sum()/len(mask):.3f}")
        print(f"Proportion of eligible pts with ONLY baseline PHQ-9 and demographics in Brighten-v1: {mask[data['study'] == 'Brighten-v1'].sum()}/{len(mask[data['study'] == 'Brighten-v1'])} = {mask[data['study'] == 'Brighten-v1'].sum()/len(mask[data['study'] == 'Brighten-v1']):.3f}")
        print(f"Proportion of eligible pts with ONLY baseline PHQ-9 and demographics in Brighten-v2: {mask[data['study'] == 'Brighten-v2'].sum()}/{len(mask[data['study'] == 'Brighten-v2'])} = {mask[data['study'] == 'Brighten-v2'].sum()/len(mask[data['study'] == 'Brighten-v2']):.3f}")

        data = data[~mask].reset_index(drop=True)

    # Exclude participants with a history of mania
    data = data[data["mania_history"] != 1].reset_index(drop=True)

    print("Total eligible participants:", len(data))
    print("Total eligible participants in Brighten-v1:", len(data[data["study"] == "Brighten-v1"]))
    print("Total eligible participants in Brighten-v2:", len(data[data["study"] == "Brighten-v2"]))


    for study in args.studies_to_include:
        print("-------------------\nSaving data for", study)
        study_data = data[data['study'] == study]

        # Save CSV file to destination directory
        csv_path = os.path.join(args.dest_dir, f"missing_formatted_{study}_all.csv")
        study_data.to_csv(csv_path, index=False)

        # Missingness plot
        missing = study_data.isnull()
        matplotlib.use('Agg')
        plt.figure(figsize=(12, 8))
        sns.heatmap(missing, cbar=False, yticklabels=False, cmap='viridis')
        plt.title(f"Missingness Plot for {study}")
        plt.xlabel('Variables')
        plt.ylabel('Observations')
        # Save plot to destination directory
        plot_path = os.path.join(args.dest_dir, f"{study}_missingness.jpg")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory

        # Create a missingness summary
        missingness = study_data.isnull().sum().sort_values(ascending=False)
        missingness_pct = 100 * missingness / len(study_data)
        missingness_table = pd.concat([missingness, missingness_pct], axis=1, keys=['Missing Values', '% Missing'])
        # Save summary to destination directory
        summary_path = os.path.join(args.dest_dir, f"{study}_missingness_summary.csv")
        missingness_table.to_csv(summary_path)
        print(f"Missingness summary saved as {summary_path}")

if __name__ == "__main__":
    main()