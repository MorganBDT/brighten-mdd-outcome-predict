import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(standardize=False, impute=False, drop=False, filename='formatted_brighten.csv', filter_cols=True):
    pt_data_all = pd.read_csv(filename)

    predictors = [
        "base_phq9",
        "gender",
        "education",
        "working",
        "marital_status",
        "race",
        "age",
        "income_satisfaction",
        "gad7_sum",
        "sds_sum",
        "alc_sum",
        "mania_history", # True if any of screen_1, screen_2, or screen_3 are true in the IMPACT questionnaire
        "psychosis_history", # screen_4 in the IMPACT questionnaire
        "study_arm_EVO",
        "study_arm_HealthTips",
        "study_arm_iPST",
    ]
    
    if filter_cols:
        if "_Imputation_" in pt_data_all.columns:
            pt_data_all = pt_data_all[predictors + ["response", "_Imputation_"]]
            pt_data_all["_Imputation_"] = pt_data_all["_Imputation_"].astype(int)
        else:
            pt_data_all = pt_data_all[predictors + ["response"]]
    
    if impute:
        for predictor in predictors:
            if predictor not in ["study_arm_EVO", "study_arm_HealthTips", "study_arm_iPST",]:
                try:
                    pt_data_all[predictor].fillna(int(pt_data_all[predictor].mean()), inplace=True)
                except:
                    print("Couldn't get non-nan median for predictor: " + predictor)
    else:
        if drop:
            pt_data_all = pt_data_all.dropna()
    
    if standardize: 
    
        non_bin_predictors = [
            "base_phq9",
            "age",
            "gad7_sum",
            "sds_sum",
            "alc_sum",
            "mins_to_sleep",
            "sleep_hours",
            "mins_awake_after_sleep",
            "impression_of_change",
        ]

        pt_data_all_non_bin = pt_data_all[non_bin_predictors]

        scaled = StandardScaler().fit_transform(pt_data_all_non_bin.to_numpy())

        scaled = pd.DataFrame(scaled, columns=non_bin_predictors)

        pt_data_all[non_bin_predictors] = scaled[non_bin_predictors]
    
    return pt_data_all, predictors