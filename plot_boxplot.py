import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
STL = pd.read_csv("output/oncotypedx_stl_learn/STL_combined_GSE20271_FAC.csv", skiprows=1)
AUX = pd.read_csv("output/oncotypedx_aux_learn/AUX_combined_GSE20271_FAC.csv")

# Add a 'Method' column to distinguish between STL and AUX
STL['Method'] = 'Single Task'
AUX['Method'] = 'Auxiliary'

# Combine the STL and AUX data into a single DataFrame
combined_df = pd.concat([STL, AUX], ignore_index=True)

# Plotting
plt.figure(figsize=(14, 6))

# AUROC
plt.subplot(1, 2, 1)
sns.boxplot(data=combined_df, x="Method", y="AUROC")
plt.title("Comparison of AUROC between Single Task and Auxiliary Learning")
plt.ylabel("AUROC")
plt.xlabel("Method")

# AUPR
plt.subplot(1, 2, 2)
sns.boxplot(data=combined_df, x="Method", y="AUPR")
plt.title("Comparison of AUPR between Single Task and Auxiliary Learning")
plt.ylabel("AUPR")
plt.xlabel("Method")

# Main Task: Brightness_TNBC_paclitaxel_AC
plt.suptitle("Main Task: Breast_GSE20271_FAC", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
