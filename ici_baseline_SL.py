# Necessary imports
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import argparse
import glob
from data_utils import BreastCancerDataset
import xgboost as xgb  # Importing XGBoost

# Initialize an empty list to store results
output_df = []

# Argument parser to take command-line arguments
parser = argparse.ArgumentParser(description='Run model with different seeds.')
parser.add_argument('--mccv', type=int, default=1, help='Seed for the random number generator')  # `required` removed for default
parser.add_argument('--main_task_file', type=str, default='GSE164458_paclitaxel.csv', help='File name for the main task')  # `required` removed for default
args = parser.parse_args()

# Collect all CSV file paths from the specified directory
file_paths_unsorted = glob.glob('./clindb_ici/*.csv')
# Ensure the "main_task_file" is first and sort the remaining ones
file_paths = sorted(file_paths_unsorted, key=lambda x: (args.main_task_file not in x, x))

# Generate immune_paths based on the file_paths
immune_paths = ['./immune_profile/immune_' + os.path.basename(fp) for fp in file_paths]

# Read the main task files
rna_data = pd.read_csv(file_paths[0])
immune_data = pd.read_csv(immune_paths[0])

# Combine gene and CIBERSORT features
# Select columns that match the genes of interest in rna_data and those starting with 'CIBERSORT' in immune_data
# gene_features = rna_data[genes_21]
immune_features = immune_data.filter(regex='^CIBERSORT')
data = immune_features

# Split into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(data, rna_data.iloc[:,1], test_size=0.3, random_state=args.mccv, stratify=rna_data.iloc[:,1])

# Parameter grids for grid search
randomforest_grid = {
    "classifier__max_depth": [3, 5, 10],
    "classifier__n_estimators": [50, 100, 200],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 5],
    "classifier__max_leaf_nodes": [None, 10, 20],
    "classifier__n_jobs": [10],
}

mlp_grid = {
    "classifier__hidden_layer_sizes": [(5,), (10,)],
    "classifier__alpha": [0.001, 0.01, 0.1],
    "classifier__activation": ["relu"],
    "classifier__solver": ["lbfgs", "adam"],
    "classifier__max_iter": [500, 1000],
    "classifier__learning_rate_init": [0.001, 0.01],
}

logisticregression_grid = {
    "classifier__penalty": ['l2'],
    "classifier__solver": ['saga'],
    "classifier__max_iter": [10000],
    "classifier__class_weight": ['balanced'],
    "classifier__C": [0.01, 0.1, 1, 10],
}

xgboost_grid = {
    "max_depth": [3, 5, 7, 9],
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# Define the preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('variance_threshold', VarianceThreshold(threshold=0)),
    # ('select_k_best', SelectKBest(f_classif, k=500)),
    ('scaler', StandardScaler()),
    # ('pca', PCA(n_components=10))
])

# Modify baseline models to use RandomizedSearchCV
baseline_models = {
    "RandomForest": RandomizedSearchCV(
        Pipeline([('preprocessing', preprocessing_pipeline), ('classifier', RandomForestClassifier())]),
        param_distributions=randomforest_grid,
        n_iter=50,
        cv=5,
        random_state=args.mccv,
        n_jobs=10
    ),
    "MLP": RandomizedSearchCV(
        Pipeline([('preprocessing', preprocessing_pipeline), ('classifier', MLPClassifier())]),
        param_distributions=mlp_grid,
        n_iter=50,
        cv=5,
        random_state=args.mccv,
        n_jobs=20
    ),
    "LogisticRegression": RandomizedSearchCV(
        Pipeline([('preprocessing', preprocessing_pipeline), ('classifier', LogisticRegression())]),
        param_distributions=logisticregression_grid,
        n_iter=50,
        cv=5,
        random_state=args.mccv,
        n_jobs=10
    ),
        "XGBoost": RandomizedSearchCV(xgb.XGBClassifier(random_state=args.mccv, use_label_encoder=False, eval_metric='logloss'), xgboost_grid, n_iter=50, cv=5, random_state=args.mccv, n_jobs=10)
}

# Loop to train and evaluate models
for model_name, model in baseline_models.items():
    model.fit(X_train, y_train)
    best_model = model.best_estimator_
    probas = best_model.predict_proba(X_test)[:, 1]
    
    # Compute AUROC and AUPR
    auroc = roc_auc_score(y_test, probas)
    aupr = average_precision_score(y_test, probas)
    
    # Store the evaluation results
    test_eval = {
        'model': model_name,
        'auroc': auroc,
        'aupr': aupr,
        'sample_size': X_train.shape[0],
        'feature_size': X_train.shape[1]
    }
    
    output_df.append(test_eval)

# Save the results to a CSV file
results_df = pd.DataFrame(output_df)
main_task_name = args.main_task_file.replace('.csv', '')
results_filename = f'./output/MCCV_ici_SL_learn_{main_task_name}_{args.mccv}.csv'
results_df.to_csv(results_filename, index=False)
