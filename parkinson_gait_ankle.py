import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import randint, uniform
from imblearn.over_sampling import SMOTE

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
    print("Starting SVC...")
    param_distributions = [
        {'kernel': ['linear'], 'C': [40, 50, 60, 70, 80, 90, 100]},
        {'kernel': ['poly'], 'C': [40, 50, 60, 70, 80, 90, 100], 'degree': [2, 3, 4]},
        {'kernel': ['rbf'], 'C': [10, 20, 30, 40, 50, 60, 70], 'gamma': ['scale', 'auto']},
        {'kernel': ['sigmoid'], 'C': [40, 50, 60, 70, 80, 90, 100], 'gamma': ['scale', 'auto']}
    ]
    svc = SVC()
    random_search = RandomizedSearchCV(svc, param_distributions, n_iter=70, cv=10, scoring='f1_weighted', random_state=42, n_jobs=4)
    random_search.fit(X_train_scaled, y_train)
    print("SVC finished.")
    return random_search.best_estimator_

# K-Nearest Neighbors Classifier
def run_knn(X_train_scaled, y_train):
    print("Starting KNN...")
    knn_classifier = KNeighborsClassifier()
    param_dist = {
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': randint(1, 50),
        'p': uniform(1, 10),
        'n_neighbors': np.arange(2, 30, 1),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    }
    random_search = RandomizedSearchCV(estimator=knn_classifier, param_distributions=param_dist, n_iter=100, cv=10, scoring='f1_weighted', random_state=0, n_jobs=4)
    random_search.fit(X_train_scaled, y_train)
    print(f"KNN  finished. Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_

# Decision Tree Classifier
def run_decision_tree(X_train_scaled, y_train):
    print("Starting Decision Tree...")
    param_dist_dt = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'max_depth': [None] + list(range(1, 32)),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 9),
        'min_weight_fraction_leaf': uniform(0, 0.5),
        'max_features': [None, 'sqrt', 'log2'] + list(range(1, 20)),  # Removed 'auto'
        'random_state': [None, 1],
        'max_leaf_nodes': [None] + list(range(2, 50)),
        'min_impurity_decrease': uniform(0, 0.2),
        'class_weight': [None, 'balanced'],
        'ccp_alpha': uniform(0, 0.1)
    }
    dt_classifier = DecisionTreeClassifier()
    random_search_dt = RandomizedSearchCV(estimator=dt_classifier, param_distributions=param_dist_dt, n_iter=100, cv=10, scoring='f1_weighted', random_state=0, n_jobs=4)
    random_search_dt.fit(X_train_scaled, y_train)
    print(f"Decision Tree finished. Best parameters: {random_search_dt.best_params_}")
    return random_search_dt.best_estimator_

# Gaussian Naive Bayes Classifier
def run_gaussian_nb(X_train_scaled, y_train):
    print("Starting Gaussian Naive Bayes...")
    nb_classifier = GaussianNB()
    param_dist = {
        'var_smoothing': np.logspace(0, -9, num=100)
    }
    random_search = RandomizedSearchCV(estimator=nb_classifier, param_distributions=param_dist, n_iter=100, cv=10, scoring='f1_weighted', n_jobs=-1, random_state=0)
    random_search.fit(X_train_scaled, y_train)
    print(f"Gaussian Naive Bayes  finished. Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_

# Bagging Classifier
def run_bagging(X_train_scaled, y_train):
    print("Starting Bagging Classifier...")
    dt_classifier = DecisionTreeClassifier()
    param_dist = {
        'n_estimators': [20, 30, 40],
        'max_features': [0.92, 0.95, 1.0],
        'max_samples': [x / 10 for x in range(1, 3)],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False],
        'warm_start': [False],
        'oob_score': [True],  # Set to True only when bootstrap=True
    }

    bc_classifier = BaggingClassifier(estimator=dt_classifier, n_jobs=-1, random_state=1)

    random_search = RandomizedSearchCV(estimator=bc_classifier, param_distributions=param_dist, n_iter=100, cv=10, random_state=1, n_jobs=4)
    random_search.fit(X_train_scaled, y_train)

    print(f"Bagging Classifier finished. Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_

# AdaBoost Classifier
def run_adaboost(X_train_scaled, y_train):
    print("Starting AdaBoost...")
    abc_classifier = AdaBoostClassifier()
    param_dist = {
        'n_estimators': [70, 80, 90, 100, 200, 300],
        'learning_rate': [0.08, 0.09, 0.1, 0.2, 0.3],
        'algorithm': ['SAMME', 'SAMME.R']
    }
    random_search = RandomizedSearchCV(estimator=abc_classifier, param_distributions=param_dist, n_iter=100, cv=10, random_state=1, n_jobs=4)
    random_search.fit(X_train_scaled, y_train)
    print(f"AdaBoost  finished. Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_

# LightGBM Classifier
def run_lightgbm(X_train_scaled, y_train):
    print("Starting LightGBM...")
    lgbm_classifier = lgb.LGBMClassifier(random_state=1)
    param_distributions = {
        'boosting_type': ['gbdt', 'dart', 'goss'],  # Focus on the most stable and commonly used boosting type
        'num_leaves': randint(10, 25),  # Reduce the range of leaves
        'max_depth': randint(2, 7),  # Limit tree depth to prevent overfitting
        'learning_rate': uniform(0.05, 0.2),  # Avoid extremely low learning rates
        'n_estimators': randint(50, 100),  # Use a smaller range for estimators
        'subsample': uniform(0.7, 1.0),  # Ensure enough samples are used in each iteration
        'colsample_bytree': uniform(0.7, 1.0),  # Ensure enough features are used
        'reg_alpha': uniform(0.0, 0.1),  # Limit regularization to prevent underfitting
        'reg_lambda': uniform(0.0, 0.1),
        'min_split_gain': uniform(0.0, 0.01),  # Allow small gain splits
    }
    random_search = RandomizedSearchCV(estimator=lgbm_classifier, param_distributions=param_distributions, n_iter=10, cv=5, random_state=1, n_jobs=-1)
    random_search.fit(X_train_scaled, y_train)
    print(f"LightGBM finished. Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_

# Random Forest Classifier
def run_random_forest(X_train_scaled, y_train):
    print("Starting Random Forest...")
    rf_classifier = RandomForestClassifier(random_state=1)
    param_distributions = {
        'bootstrap': [True, False],
        'max_depth': [10, 15, 20, 25, 30],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [2, 4, 6],
        'min_samples_split': [5, 10, 15],
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy']
    }
    random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_distributions, n_iter=100, cv=10, scoring='f1_weighted', n_jobs=4, random_state=1)
    random_search.fit(X_train_scaled, y_train)
    print(f"Random Forest  finished. Best parameters: {random_search.best_params_}")
    return random_search
def run_catboost(X_train_scaled, y_train):
    print("Starting CatBoost...")
    catboost_classifier = CatBoostClassifier(random_state=1, verbose=False)
    param_distributions = {
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 4, 5],
        'n_estimators': [10, 20, 30]
    }
    random_search = RandomizedSearchCV(estimator=catboost_classifier, param_distributions=param_distributions, n_iter=100, cv=10, scoring='f1_weighted', n_jobs=4, random_state=1)
    random_search.fit(X_train_scaled, y_train)
    print(f"CatBoost  finished. Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_
# XGBoost Classifier
def run_xgboost(X_train_scaled, y_train):
    print("Starting XGBoost...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss')
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [10, 20, 30, 40],
        'colsample_bytree': [0.5, 0.7, 1],
        'subsample': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 4, 5]
    }
    random_search = RandomizedSearchCV(estimator=xgb_clf, param_distributions=param_grid, n_iter=100, scoring='f1_weighted', random_state=42, n_jobs=4, cv=10)
    random_search.fit(X_train_scaled, y_encoded)
    xgb_clf = random_search.best_estimator_
    xgb_clf.fit(X_train_scaled, y_encoded)
    print(f"XGBoost  finished. Best parameters: {random_search.best_params_}")
    return xgb_clf

def evaluate_overfitting(rf_classifier, X_train_scaled, y_train, X_test_scaled, y_test, best_params):
    # Check for overfitting
    train_accuracy = rf_classifier.score(X_train_scaled, y_train)
    test_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test_scaled))
    accuracy_diff = train_accuracy - test_accuracy

    print("Checking for overfitting based on accuracy...")
    if accuracy_diff > 0.1:  # Set threshold for overfitting detection
        print(f"Warning: Possible overfitting detected! Difference = {accuracy_diff:.4f}")

    train_f1 = f1_score(y_train, rf_classifier.predict(X_train_scaled), average='weighted')
    test_f1 = f1_score(y_test, rf_classifier.predict(X_test_scaled), average='weighted')
    f1_diff = train_f1 - test_f1

    print("Checking for overfitting based on F1 score...")
    if f1_diff > 0.1:  # Set threshold for overfitting detection
        print(f"Warning: Possible overfitting detected! Difference = {f1_diff:.4f}")

    # Plot the amount of overfitting by showing training and testing error over the training period

    # Initialize the classifier with only one tree to start with
    rf_classifier = RandomForestClassifier(
        n_estimators=1, 
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_leaf=best_params['min_samples_leaf'],
        min_samples_split=best_params['min_samples_split'],
        criterion=best_params['criterion'],
        bootstrap=best_params['bootstrap'],
        random_state=1,
        warm_start=True  # Allows adding more trees without starting from scratch
    )

    n_estimators_range = np.arange(1, 201)  # Range for number of trees
    train_errors = []
    test_errors = []

    for n_estimators in n_estimators_range:
        rf_classifier.n_estimators = n_estimators
        rf_classifier.fit(X_train_scaled, y_train)
        
        # Calculate errors
        y_train_pred = rf_classifier.predict(X_train_scaled)
        y_test_pred = rf_classifier.predict(X_test_scaled)
        
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        train_errors.append(1 - train_f1)
        test_errors.append(1 - test_f1)

    # Plot the errors
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_errors, label='Training Error (1 - F1 Score)', color='blue')
    plt.plot(n_estimators_range, test_errors, label='Testing Error (1 - F1 Score)', color='red')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Error (1 - F1 Score)')
    plt.title('Training vs. Testing Error as a Function of Number of Estimators')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    # Example usage:
    input_directory = "./data/GaitCyclesFormattedCSVfiles"
    target_directory = "./data/"
    target_feature = 'Ankle'


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

    # evaluate_overfitting(rf_best.best_estimator_, X_train_scaled, y_train, X_test_scaled, y_test, rf_best.best_params_)

    metric_printer(catboost_best, X_test_scaled, y_test)

    metric_printer(xgb_best, X_test_scaled, y_test)

