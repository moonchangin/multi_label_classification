import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
SL = pd.read_csv("output/ici_SL_learn/SL_combined_Gide_Cell_2019_pembro_ipi.csv", skiprows=1)
STL = pd.read_csv("output/ici_stl_learn/STL_combined_Gide_Cell_2019_pembro_ipi.csv", skiprows=1)
ADALOSS = pd.read_csv("output/ici_adaloss/adaloss_combined_Gide_Cell_2019_pembro_ipi.csv", skiprows=1)
GRADNORM = pd.read_csv("output/ici_gradnorm/gradnorm_combined_Gide_Cell_2019_pembro_ipi.csv", skiprows=1)
OL_AUX = pd.read_csv("output/ici_ol_aux/ol_aux_combined_Gide_Cell_2019_pembro_ipi.csv", skiprows=1)
ARML = pd.read_csv("output/ici_aux_learn/AUX_combined_Gide_Cell_2019_pembro_ipi.csv", skiprows=1)


# Add a 'Method' column to distinguish between STL and AUX
STL['Method'] = 'STL'
ADALOSS['Method'] = 'ADALOSS'
GRADNORM['Method'] = 'GRADNORM'
OL_AUX['Method'] = 'OL_AUX'
ARML['Method'] = 'ARML'

# process for STL
XGBOOST = SL[SL['model'] == 'XGBoost']
RandomForest =  SL[SL['model'] == 'RandomForest']
MLP =   SL[SL['model'] == 'MLP']
LogisticRegression = SL[SL['model'] == 'LogisticRegression']

# combine the STL data
SL = pd.concat([XGBOOST, RandomForest, MLP, LogisticRegression], ignore_index=True)
# use 1st to3rd column for model, AUC and AUPR
SL = SL.iloc[:, [0, 1, 2]]
# rename the columns to auroc to AUC, aupr to AUPR
SL.columns = ['Method','AUROC', 'AUPR']

# Combine the all data into a single DataFrame
combined_df = pd.concat([SL, STL, ADALOSS, GRADNORM, OL_AUX, ARML], ignore_index=True)

# combined_df = pd.concat([STL, ARML], ignore_index=True)

# Plotting
plt.figure(figsize=(14, 6))

# AUROC
plt.subplot(1, 2, 1)
sns.boxplot(data=combined_df, x="Method", y="AUROC")
plt.title("Gide_Cell_2019_pembro_ipi AUROCs across 30 MCCV")
plt.ylabel("AUROC")
plt.xlabel("Method")
plt.xticks(rotation=45)  # Rotate by 45 degrees, adjust as needed
# Add text on the right side
details = """
Subtype: HER2
Treatment: paclitaxel, doxorubicin, cyclophosphamide, trebananib, trastuzumab
Response Evaluation: pathologic
Sample Size: 19
Responder Size: 6
Non-Responder Size: 13
"""
# AUPR
# plt.subplot(1, 2, 2)
# sns.boxplot(data=combined_df, x="Method", y="AUPR")
# plt.title("GSE194040_TAC_AMG386_H AUPRs across 30 MCCV")
# plt.ylabel("AUPR")
# plt.xlabel("Method")

# Main Task: Brightness_TNBC_paclitaxel_AC
# plt.suptitle("Main Task: Breast_GSE20271_FAC", fontsize=14, fontweight='bold')

plt.tight_layout()

# save the plot as pdf
plt.savefig("output/boxplot_Gide_Cell_2019_pembro_ipi.pdf")

# significance table
import pandas as pd
import scipy.stats as stats
from itertools import combinations

# Create a DataFrame for the AUROC scores
auroc_scores = combined_df.groupby("Method")["AUROC"].apply(list)

# Create an empty DataFrame to store p-values
methods = auroc_scores.index
p_value_table = pd.DataFrame(index=methods, columns=methods)

# Perform pairwise t-tests for each unique pair of methods
for method1, method2 in combinations(methods, 2):
    # Perform a t-test
    t_stat, p_value = stats.ttest_ind(auroc_scores[method1], auroc_scores[method2], equal_var=False)  # Welch's t-test
    
    # Store p-values in the table
    p_value_table.at[method1, method2] = p_value
    p_value_table.at[method2, method1] = p_value  # Symmetry

# Fill diagonal with NaN or other indicator
p_value_table.fillna("NA", inplace=True)

# Save the table as a CSV file
p_value_table.to_csv("output/pairwise_ttest_Gide_Cell_2019_pembro_ipi.csv")
