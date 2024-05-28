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
adaloss_prefix = "adaloss_combined_"
gradnorm_prefix = "gradnorm_combined_"
ol_aux_prefix = "ol_aux_combined_"

# Calculate summaries for all learning types
aux_learn_summary = calculate_summary("./output/oncotypedx_aux_learn", "aux_learn", aux_prefix)
stl_learn_summary = calculate_summary("./output/oncotypedx_stl_learn", "stl_learn", stl_prefix)
adaloss_learn_summary = calculate_summary("./output/oncotypedx_adaloss", "adaloss_learn", adaloss_prefix)
gradnorm_learn_summary = calculate_summary("./output/oncotypedx_gradnorm", "gradnorm_learn", gradnorm_prefix)
ol_aux_learn_summary = calculate_summary("./output/oncotypedx_ol_aux", "ol_aux_learn", ol_aux_prefix)

# # Merge summaries and restrict to common studies
# common_studies = set(gradnorm_learn_summary['Study']).intersection(ol_aux_learn_summary['Study'])
# filtered_aux = aux_learn_summary[aux_learn_summary['Study'].isin(common_studies)]
# filtered_stl = stl_learn_summary[stl_learn_summary['Study'].isin(common_studies)]
# filtered_adaloss = adaloss_learn_summary[adaloss_learn_summary['Study'].isin(common_studies)]

merged_summary = pd.merge(aux_learn_summary, stl_learn_summary, on="Study", how='outer')
merged_summary = pd.merge(merged_summary, adaloss_learn_summary, on="Study", how='outer')
merged_summary = pd.merge(merged_summary, gradnorm_learn_summary, on="Study", how='outer')
final_summary = pd.merge(merged_summary, ol_aux_learn_summary, on="Study", how='outer')

# Sort the DataFrame based on a relevant column, here using 'Study'
final_summary_sorted = final_summary.sort_values(by='Study')

# Save the merged summary to a csv file
final_summary_sorted.to_csv("output/summary/oncotypedx_complete_summary.csv", index=False)

print(final_summary_sorted)
