import pandas as pd
import os

def calculate_summary(folder_path, learning_type, prefix):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
    summaries = []
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        try:
            auc_learn = pd.read_csv(csv_file_path, skiprows=1)
            if auc_learn.empty:
                raise ValueError("Empty DataFrame")
            auc_mean = auc_learn.iloc[:, 0].mean()
            auc_std = auc_learn.iloc[:, 0].std()
            aupr_mean = auc_learn.iloc[:, 1].mean()
            aupr_std = auc_learn.iloc[:, 1].std()
        except (pd.errors.EmptyDataError, ValueError):
            auc_mean = auc_std = aupr_mean = aupr_std = None
        
        study_name = csv_file.replace(prefix, "").replace(".csv", "")
        
        summaries.append({
            "Study": study_name,
            f"{learning_type}_AUC_Mean": auc_mean,
            f"{learning_type}_AUC_Std": auc_std,
            f"{learning_type}_AUPR_Mean": aupr_mean,
            f"{learning_type}_AUPR_Std": aupr_std
        })
    return pd.DataFrame(summaries)

def calculate_SL_summary(folder_path, learning_type, prefix):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
    summaries = []
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        try:
            auc_learn = pd.read_csv(csv_file_path, skiprows=1)
            if auc_learn.empty:
                raise ValueError("Empty DataFrame")
            model_auc = auc_learn[auc_learn['model'] == learning_type]
            if model_auc.empty:
                raise ValueError("Model type not found")
            auc_mean = model_auc.iloc[:, 1].mean()
            auc_std = model_auc.iloc[:, 1].std()
            aupr_mean = model_auc.iloc[:, 2].mean()
            aupr_std = model_auc.iloc[:, 2].std()
        except (pd.errors.EmptyDataError, ValueError):
            auc_mean = auc_std = aupr_mean = aupr_std = None

        study_name = csv_file.replace(prefix, "").replace(".csv", "")
        summaries.append({
            "Study": study_name,
            f"{learning_type}_AUC_Mean": auc_mean,
            f"{learning_type}_AUC_Std": auc_std,
            f"{learning_type}_AUPR_Mean": aupr_mean,
            f"{learning_type}_AUPR_Std": aupr_std
        })
    return pd.DataFrame(summaries)

# Prefixes used in the study names
stl_prefix = "STL_combined_"
aux_prefix = "AUX_combined_"
adaloss_prefix = "adaloss_combined_"
gradnorm_prefix = "gradnorm_combined_"
ol_aux_prefix = "ol_aux_combined_"
SL_prefix = "SL_combined_"


# Calculate summaries for all learning types
aux_learn_summary = calculate_summary("./output/ici_aux_learn", "ARML", aux_prefix)
stl_learn_summary = calculate_summary("./output/ici_stl_learn", "STL", stl_prefix)
adaloss_learn_summary = calculate_summary("./output/ici_adaloss", "ADALOSS", adaloss_prefix)
gradnorm_learn_summary = calculate_summary("./output/ici_gradnorm", "GRADNORM", gradnorm_prefix)
ol_aux_learn_summary = calculate_summary("./output/ici_ol_aux", "OL_AUX", ol_aux_prefix)

# Calculate supervised learning summaries
XGBoost_learn_summary = calculate_SL_summary("./output/ici_SL_learn", "XGBoost", SL_prefix)
RandomForest_learn_summary = calculate_SL_summary("./output/ici_SL_learn", "RandomForest", SL_prefix)
MLP_learn_summary = calculate_SL_summary("./output/ici_SL_learn", "MLP", SL_prefix)
LogisticRegression_learn_summary = calculate_SL_summary("./output/ici_SL_learn", "LogisticRegression", SL_prefix)

# Merge summaries and restrict to common studies
common_studies = set(gradnorm_learn_summary['Study']).intersection(ol_aux_learn_summary['Study'])

filtered_aux = aux_learn_summary[aux_learn_summary['Study'].isin(common_studies)]
filtered_stl = stl_learn_summary[aux_learn_summary['Study'].isin(common_studies)]
filtered_adaloss = adaloss_learn_summary[aux_learn_summary['Study'].isin(common_studies)]

# Start merging with new SL summaries
merged_summary = pd.merge(filtered_aux, filtered_stl, on="Study", how='outer')
merged_summary = pd.merge(merged_summary, filtered_adaloss, on="Study", how='outer')
merged_summary = pd.merge(merged_summary, gradnorm_learn_summary, on="Study", how='outer')
merged_summary = pd.merge(merged_summary, ol_aux_learn_summary, on="Study", how='outer')

# Merge supervised learning summaries
merged_summary = pd.merge(merged_summary, XGBoost_learn_summary, on="Study", how='outer')
merged_summary = pd.merge(merged_summary, RandomForest_learn_summary, on="Study", how='outer')
merged_summary = pd.merge(merged_summary, MLP_learn_summary, on="Study", how='outer')
merged_summary = pd.merge(merged_summary, LogisticRegression_learn_summary, on="Study", how='outer')

# Sort the DataFrame based on 'Study'
final_summary_sorted = merged_summary.sort_values(by='Study')

# Save the merged summary to a CSV file
final_summary_sorted.to_csv("output/summary/tmp_ici_complete_summary_manuscript.csv", index=False)

# Print the final summary
print(final_summary_sorted)

