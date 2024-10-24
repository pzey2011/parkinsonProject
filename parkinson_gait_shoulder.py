import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import optuna
from sklearn.model_selection import cross_val_score
from optuna.samplers import RandomSampler

# Function to check if the file should be used for training
def is_training_file(filename):
    if "02_off" in filename or "15_off" in filename:
        return False
    sub_id = int(filename[3:5])
    return 1 <= sub_id <= 18 or sub_id == 20

# Function to extract the target from the file
def get_target_from_file(filename, target_dfs):
    collected_data = pd.DataFrame()
    for sheet_name, sheet_df in target_dfs.items():
        id = filename.split('_')[0]
        target_column = 'OFF - Hoehn & Yahr ' if 'off' in filename else 'ON - Hoehn & Yahr ' if 'on' in filename else None
        if target_column is None:
            continue
        if id in sheet_df['ID'].values:
            target_row = sheet_df.loc[sheet_df['ID'] == id, ['ID', target_column]]
            collected_data = pd.concat([collected_data, target_row], ignore_index=True)
    return collected_data

# Function to load data and prepare the dataset
def load_data(input_directory, target_directory, target_feature):
    # Initialize empty DataFrames to hold the final results
    right_df = pd.DataFrame()
    left_df = pd.DataFrame()

    # Load target data from Excel file
    target_df = None
    for target_filename in os.listdir(target_directory):
        if target_filename.endswith(".xlsx"):
            target_file_path = os.path.join(target_directory, target_filename)
            target_df = pd.read_excel(target_file_path, sheet_name=None)
            break

    if target_df is None:
        print("No target file found.")
        return None, None, None, None

    # Process each CSV file in the input directory
    for input_filename in os.listdir(input_directory):
        if input_filename.endswith(".csv") and target_feature in input_filename and 'SUB17_on' not in input_filename:
            input_file_path = os.path.join(input_directory, input_filename)
            try:
                input_numerical_data = pd.DataFrame()
                input_specific_df = pd.read_csv(input_file_path)

                # Convert all columns to numeric, coerce errors to NaN, and drop rows with NaN values
                input_numerical_data = input_specific_df.apply(pd.to_numeric, errors='coerce')
                input_numerical_data.dropna(inplace=True)
                input_numerical_data.reset_index(drop=True, inplace=True)

                if input_numerical_data.isna().any().any() or input_numerical_data.empty:
                    print(f"Skipping file {input_filename} due to insufficient data after cleaning.")
                    continue

                if 'Gait cycle [%]' in input_numerical_data.columns:
                    input_numerical_data.drop('Gait cycle [%]', axis=1, inplace=True)

                input_numerical_data['is_training'] = pd.Series(
                    [is_training_file(input_filename)] * len(input_numerical_data))
                # Get the related target data using the file name
                related_target = get_target_from_file(input_filename, target_df)
                if related_target is not None:
                    target_column = 'OFF - Hoehn & Yahr ' if 'off' in input_filename else 'ON - Hoehn & Yahr ' if 'on' in input_filename else None
                    if target_column and target_column in related_target.columns:
                        target_data = related_target[target_column].iloc[0]
                        target_series = pd.Series([target_data] * len(input_numerical_data))
                        input_numerical_data['target'] = target_series
                    if 'Right_' + target_feature in input_filename:
                        right_df = pd.concat([right_df, input_numerical_data]).reset_index(drop=True)
                    elif 'Left_' + target_feature in input_filename:
                        left_df = pd.concat([left_df, input_numerical_data]).reset_index(drop=True)

            except Exception as e:
                print(f"Error occurred while processing file {input_filename}: {e}")

    if right_df.isnull().values.any():
        # If there are NaN values, execute this block
        right_df = right_df.dropna(axis=1, how='any')
    if left_df.isnull().values.any():
        # If there are NaN values, execute this block
        left_df = left_df.dropna(axis=1, how='any')
    common_columns = right_df.columns.intersection(left_df.columns)
    columns_to_exclude = ['is_training', 'target']
    columns_to_rename = [col for col in common_columns if col not in columns_to_exclude]
    right_df = right_df.rename(columns={col: 'Right_' + col for col in columns_to_rename})
    left_df = left_df.rename(columns={col: 'Left_' + col for col in columns_to_rename})
    left_df = left_df.drop(columns=columns_to_exclude)

    all_data = pd.concat([right_df.reset_index(drop=True), left_df.reset_index(drop=True)], axis=1)

    # Handle NaN values by trimming the DataFrame
    if all_data.isna().any().any():
        first_nan_row = all_data.isna().any(axis=1).idxmax()
        all_data_trimmed = all_data.iloc[:first_nan_row, :]
    else:
        all_data_trimmed = all_data.copy()

    all_data = all_data_trimmed.copy()

    # Extract training and testing data based on 'is_training' flag
    training_all_data = all_data[all_data['is_training']].drop('is_training', axis=1)
    test_all_data = all_data[~all_data['is_training']].drop('is_training', axis=1)

    # Extract the target data
    training_all_Y = training_all_data['target']
    test_all_Y = test_all_data['target']

    # Extract the input data
    training_all_X = training_all_data.drop('target', axis=1)
    test_all_X = test_all_data.drop('target', axis=1)

    # Reset index
    test_all_X = test_all_X.reset_index(drop=True)
    test_all_Y = test_all_Y.reset_index(drop=True)
    training_all_X = training_all_X.reset_index(drop=True)
    training_all_Y = training_all_Y.reset_index(drop=True)

    # For the training set
    X_train, y_train = shuffle(training_all_X, training_all_Y, random_state=42)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # For the test set
    X_test, y_test = shuffle(test_all_X, test_all_Y, random_state=42)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Oversample X_train, y_train
    smote = SMOTE(random_state=42)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
    print("Data preprocessing finished.")
    return X_train_scaled, X_test_scaled, y_train, y_test


# Assuming input data preparation and preprocessing have been done as shown in the previous code

# Define the metric_printer function
def metric_printer(classifier, X_test_scaled, y_test):
    classifier_name = classifier.__class__.__name__
    if classifier_name == 'XGBClassifier':
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)
        y_pred = classifier.predict(X_test_scaled)
        y_pred = label_encoder.inverse_transform(y_pred)
    else:
        y_pred = classifier.predict(X_test_scaled)
    print(f"Classifier: {classifier_name}")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, zero_division=0, average='weighted')
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print('____________________________________')
    results[classifier_name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Support Vector Machine Classifier
def run_svc(X_train_scaled, y_train):
    print("Starting SVC with Optuna...")

    def objective(trial, X_train_scaled, y_train):
        # Suggest hyperparameters to be optimized
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        C = trial.suggest_int('C', 1, 100)

        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 4)
        else:
            degree = 3  # Default for other kernels

        if kernel in ['rbf', 'sigmoid']:
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        else:
            gamma = 'scale'  # Default for kernels that don't need gamma

        # Create an SVC model with the selected hyperparameters
        svc = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)

        # Split the data for quick early stopping check (first 30% of data)
        X_partial, y_partial = X_train_scaled[:int(0.3 * len(X_train_scaled))], y_train[:int(0.3 * len(y_train))]

        # Perform a quick 3-fold cross-validation on a subset of the data for early stopping
        partial_scores = cross_val_score(svc, X_partial, y_partial, cv=3, scoring='f1_weighted', n_jobs=-1)
        if np.mean(partial_scores) < 0.5:  # Threshold for early stopping
            raise optuna.exceptions.TrialPruned()

        # Full 3-fold cross-validation for remaining trials
        scores = cross_val_score(svc, X_train_scaled, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)

        return scores.mean()

    study = optuna.create_study(direction='maximize', sampler=RandomSampler(seed=42),
                                pruner=optuna.pruners.MedianPruner())

    # Optimize the study with pruning enabled
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train), n_trials=15)

    print("Optuna finished.")

    # Get the best hyperparameters
    best_params = study.best_params

    # Train a final SVC model using the best hyperparameters
    best_svc = SVC(**best_params)
    best_svc.fit(X_train_scaled, y_train)

    return best_svc
# K-Nearest Neighbors Classifier
def run_knn(X_train_scaled, y_train):
    def objective_knn(trial):
        knn_classifier = KNeighborsClassifier(
            n_neighbors=trial.suggest_int('n_neighbors', 2, 30),
            algorithm=trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            leaf_size=trial.suggest_int('leaf_size', 1, 50),
            p=trial.suggest_float('p', 1, 10),
            weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
            metric=trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
        )
        score = cross_val_score(knn_classifier, X_train_scaled, y_train, cv=5, scoring='f1_weighted').mean()
        return score

    print("Starting KNN with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_knn, n_trials=30)
    print(f"KNN finished. Best parameters: {study.best_params}")
    best_knn = KNeighborsClassifier(**study.best_params)
    best_knn.fit(X_train_scaled, y_train)
    return best_knn

# Decision Tree Classifier
def run_decision_tree(X_train_scaled, y_train):
    def objective_dt(trial):
        dt_classifier = DecisionTreeClassifier(
            criterion=trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            splitter=trial.suggest_categorical('splitter', ['best', 'random']),
            max_depth=trial.suggest_int('max_depth', 1, 32),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 11),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 9),
            min_weight_fraction_leaf=trial.suggest_float('min_weight_fraction_leaf', 0, 0.5),
            max_features=trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']),
            max_leaf_nodes=trial.suggest_int('max_leaf_nodes', 2, 50),
            min_impurity_decrease=trial.suggest_float('min_impurity_decrease', 0, 0.2),
            class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
            ccp_alpha=trial.suggest_float('ccp_alpha', 0, 0.1)
        )

        score = cross_val_score(dt_classifier, X_train_scaled, y_train, cv=10, scoring='f1_weighted').mean()
        return score

    print("Starting Decision Tree with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_dt, n_trials=50)
    print(f"Decision Tree finished. Best parameters: {study.best_params}")
    best_dt = DecisionTreeClassifier(**study.best_params)
    best_dt.fit(X_train_scaled, y_train)
    return best_dt

# Gaussian Naive Bayes Classifier
def run_gaussian_nb(X_train_scaled, y_train):
    def objective_nb(trial):
        nb_classifier = GaussianNB(
            var_smoothing=trial.suggest_float('var_smoothing', 1e-9, 1e0, log=True)
        )

        score = cross_val_score(nb_classifier, X_train_scaled, y_train, cv=10, scoring='f1_weighted').mean()
        return score

    print("Starting Gaussian Naive Bayes with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_nb, n_trials=50)
    print(f"Gaussian Naive Bayes finished. Best parameters: {study.best_params}")
    best_nb = GaussianNB(**study.best_params)
    best_nb.fit(X_train_scaled, y_train)
    return best_nb

# Bagging Classifier
def run_bagging(X_train_scaled, y_train):
    def objective_bagging(trial):
        dt_classifier = DecisionTreeClassifier()

        # Suggest hyperparameters for BaggingClassifier
        n_estimators = trial.suggest_int('n_estimators', 20, 40)
        max_features = trial.suggest_float('max_features', 0.92, 1.0)
        max_samples = trial.suggest_float('max_samples', 0.1, 1.0)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        bootstrap_features = trial.suggest_categorical('bootstrap_features', [True, False])

        # Ensure oob_score is only True if bootstrap is True
        oob_score = True if bootstrap else False

        bc_classifier = BaggingClassifier(
            estimator=dt_classifier,
            n_estimators=n_estimators,
            max_features=max_features,
            max_samples=max_samples,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score
        )
        score = cross_val_score(bc_classifier, X_train_scaled, y_train, cv=5, scoring='f1_weighted').mean()
        return score

    print("Starting Bagging with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_bagging, n_trials=15)
    print(f"Bagging finished. Best parameters: {study.best_params}")
    best_bagging = BaggingClassifier(**study.best_params)
    best_bagging.fit(X_train_scaled, y_train)
    return best_bagging

# AdaBoost Classifier
def run_adaboost(X_train_scaled, y_train):
    def objective_adaboost(trial):
        abc_classifier = AdaBoostClassifier(
            n_estimators=trial.suggest_int('n_estimators', 70, 300),
            learning_rate=trial.suggest_float('learning_rate', 0.08, 0.3),
            algorithm=trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
        )

        score = cross_val_score(abc_classifier, X_train_scaled, y_train, cv=5, scoring='f1_weighted').mean()
        return score

    print("Starting AdaBoost with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_adaboost, n_trials=15)
    print(f"AdaBoost finished. Best parameters: {study.best_params}")
    best_adaboost = AdaBoostClassifier(**study.best_params)
    best_adaboost.fit(X_train_scaled, y_train)
    return best_adaboost

# LightGBM Classifier
def run_lightgbm(X_train_scaled, y_train):
    def objective_lgbm(trial):
        lgbm_classifier = lgb.LGBMClassifier(
            boosting_type=trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            num_leaves=trial.suggest_int('num_leaves', 10, 25),
            max_depth=trial.suggest_int('max_depth', 2, 7),
            learning_rate=trial.suggest_float('learning_rate', 0.05, 0.2),
            n_estimators=trial.suggest_int('n_estimators', 50, 100),
            subsample=trial.suggest_float('subsample', 0.7, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.7, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 0.0, 0.1),
            reg_lambda=trial.suggest_float('reg_lambda', 0.0, 0.1),
            min_split_gain=trial.suggest_float('min_split_gain', 0.0, 0.01),
            random_state=1
        )
        score = cross_val_score(lgbm_classifier, X_train_scaled, y_train, cv=5, scoring='f1_weighted').mean()
        return score

    print("Starting LightGBM with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_lgbm, n_trials=15)
    print(f"LightGBM finished. Best parameters: {study.best_params}")
    best_lgbm = lgb.LGBMClassifier(**study.best_params)
    best_lgbm.fit(X_train_scaled, y_train)
    return best_lgbm

# Random Forest Classifier
def run_random_forest(X_train_scaled, y_train):
    def objective_rf(trial):
        rf_classifier = RandomForestClassifier(
            bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
            max_depth=trial.suggest_int('max_depth', 10, 30),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 2, 6),
            min_samples_split=trial.suggest_int('min_samples_split', 5, 15),
            n_estimators=trial.suggest_int('n_estimators', 50, 200),
            criterion=trial.suggest_categorical('criterion', ['gini', 'entropy']),
            random_state=1
        )
        score = cross_val_score(rf_classifier, X_train_scaled, y_train, cv=5, scoring='f1_weighted').mean()
        return score

    print("Starting Random Forest with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_rf, n_trials=15)
    print(f"Random Forest finished. Best parameters: {study.best_params}")
    best_rf = RandomForestClassifier(**study.best_params)
    best_rf.fit(X_train_scaled, y_train)
    return best_rf

# CatBoost Classifier
def run_catboost(X_train_scaled, y_train):
    def objective_catboost(trial):
        catboost_classifier = CatBoostClassifier(
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.5),
            max_depth=trial.suggest_int('max_depth', 3, 5),
            n_estimators=trial.suggest_int('n_estimators', 10, 30),
            random_state=1,
            verbose=False
        )
        score = cross_val_score(catboost_classifier, X_train_scaled, y_train, cv=10, scoring='f1_weighted').mean()
        return score

    print("Starting CatBoost with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_catboost, n_trials=30)
    print(f"CatBoost finished. Best parameters: {study.best_params}")
    best_catboost = CatBoostClassifier(**study.best_params)
    best_catboost.fit(X_train_scaled, y_train)
    return best_catboost

# XGBoost Classifier
def run_xgboost(X_train_scaled, y_train):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    def objective_xgb(trial):
        xgb_clf = xgb.XGBClassifier(
            max_depth=trial.suggest_int('max_depth', 3, 7),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
            n_estimators=trial.suggest_int('n_estimators', 10, 40),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            min_child_weight=trial.suggest_int('min_child_weight', 1, 5),
            eval_metric='mlogloss',
            random_state=42
        )
        score = cross_val_score(xgb_clf, X_train_scaled, y_train_encoded, cv=10, scoring='f1_weighted').mean()
        return score

    print("Starting XGBoost with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_xgb, n_trials=30)
    print(f"XGBoost finished. Best parameters: {study.best_params}")
    best_xgb = xgb.XGBClassifier(**study.best_params, eval_metric='mlogloss', random_state=42)
    best_xgb.fit(X_train_scaled, y_train_encoded)
    return best_xgb
if __name__ == "__main__":
    # Example usage:
    input_directory = "./data/GaitCyclesFormattedCSVfiles"
    target_directory = "./data/"
    target_feature = 'Shoulder'


    # Load and preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test = load_data(input_directory, target_directory, target_feature)

    # Dictionary to store results
    results = {}

    # Execute and evaluate each classifier
    svc_best = run_svc(X_train_scaled, y_train)

    knn_best = run_knn(X_train_scaled, y_train)

    dt_best = run_decision_tree(X_train_scaled, y_train)

    nb_best = run_gaussian_nb(X_train_scaled, y_train)

    bc_best = run_bagging(X_train_scaled, y_train)

    abc_best = run_adaboost(X_train_scaled, y_train)

    lgbm_best = run_lightgbm(X_train_scaled, y_train)

    rf_best = run_random_forest(X_train_scaled, y_train)

    catboost_best = run_catboost(X_train_scaled, y_train)

    xgb_best = run_xgboost(X_train_scaled, y_train)

    metric_printer(svc_best, X_test_scaled, y_test)

    metric_printer(knn_best, X_test_scaled, y_test)

    metric_printer(dt_best, X_test_scaled, y_test)

    metric_printer(nb_best, X_test_scaled, y_test)

    metric_printer(bc_best, X_test_scaled, y_test)

    metric_printer(abc_best, X_test_scaled, y_test)

    metric_printer(lgbm_best, X_test_scaled, y_test)

    metric_printer(rf_best, X_test_scaled, y_test)

    metric_printer(catboost_best, X_test_scaled, y_test)

    metric_printer(xgb_best, X_test_scaled, y_test)

