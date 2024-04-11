import pandas as pd
import os

def calculate_summary(folder_path, learning_type, prefix):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
    summaries = []
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        auc_learn = pd.read_csv(csv_file_path, skiprows=1)
        
        auc_mean = auc_learn.iloc[:, 0].mean()
        auc_std = auc_learn.iloc[:, 0].std()
        aupr_mean = auc_learn.iloc[:, 1].mean()
        aupr_std = auc_learn.iloc[:, 1].std()
        
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

# Calculate summaries for both learning types
aux_learn_summary = calculate_summary("./output/oncotypedx_aux_learn", "aux_learn", aux_prefix)
stl_learn_summary = calculate_summary("./output/oncotypedx_stl_learn", "stl_learn", stl_prefix)

# Merge summaries on the 'Study' column to compare
final_summary = pd.merge(aux_learn_summary, stl_learn_summary, on="Study", how='outer')

# Calculate the difference between aux_learn_AUC_Mean and stl_learn_AUC_Mean
final_summary['AUC_Mean_Difference'] = final_summary['aux_learn_AUC_Mean'] - final_summary['stl_learn_AUC_Mean']

# Sort the DataFrame based on the difference, in descending order
final_summary_sorted = final_summary.sort_values(by='AUC_Mean_Difference', ascending=False)

print(final_summary_sorted)

# save the summary to a csv file
final_summary_sorted.to_csv("output/oncotypedx_stl_v_aux_summary.csv", index=False)
