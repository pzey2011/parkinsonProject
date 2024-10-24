import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna
from imblearn.over_sampling import SMOTE


optimized_classifiers = {}
def get_target_from_file(filename, target_dfs):
    collected_data = pd.DataFrame()
    for sheet_name, sheet_df in target_dfs.items():
        id = filename.split('_')[0]
        target_column = 'OFF - Hoehn & Yahr ' if 'off' in filename else 'ON - Hoehn & Yahr ' if 'on' in filename else None
        if target_column is None:
            print(f"Filename {filename} does not specify 'on' or 'off' status.")
            continue
        if id in sheet_df['ID'].values:
            target_row = sheet_df.loc[sheet_df['ID'] == id, ['ID', target_column]]
            if not target_row.empty:
                collected_data = pd.concat([collected_data, target_row], ignore_index=True)
    return collected_data

def is_training_file(filename):
    sub_id = int(filename[3:5])
    sub_mode = filename[6:8]
    if sub_id == 2 or (sub_id == 15 and sub_mode == 'on'):
        return False
    return 1 <= sub_id <= 19

def load_data():
    c3d_train_all_X = pd.DataFrame()
    c3d_train_all_y = pd.Series(dtype='float64')
    c3d_test_all_X = pd.DataFrame()
    c3d_test_all_y = pd.Series(dtype='float64')
    input_c3d_directory = './data/C3DFormattedCSVfiles'
    target_c3d_column_keyword = 'Pelvic'

    for csv_c3d_file_path in glob.glob(os.path.join(input_c3d_directory, '**', '*.csv'), recursive=True):
        input_csv_c3d_filename = os.path.basename(csv_c3d_file_path)
        if any(walk in input_csv_c3d_filename for walk in ['walk_1_angular_kinematics','walk_2_angular_kinematics','walk_3_angular_kinematics']):
            try:
                input_c3d_df = pd.read_csv(csv_c3d_file_path)
                input_c3d_df['Time'] = pd.to_numeric(input_c3d_df['Time'], errors='coerce')
                input_c3d_df = input_c3d_df.reset_index(drop=True)
                pelvic_columns = [col for col in input_c3d_df.columns if target_c3d_column_keyword in col]
                if pelvic_columns:
                    input_pelvic_c3d_data = input_c3d_df[pelvic_columns]
                    input_pelvic_c3d_data = input_pelvic_c3d_data.drop(input_pelvic_c3d_data.index[0])
            except pd.errors.EmptyDataError:
                print(f"No data in file {input_csv_c3d_filename}, skipping.")
            except Exception as e:
                print(f"An error occurred while processing file {input_csv_c3d_filename}: {e}")

            try:
                target_directory = "./data/"
                for target_filename in os.listdir(target_directory):
                    if target_filename.endswith(".xlsx"):
                        target_file_path = os.path.join(target_directory, target_filename)
                        target_df = pd.read_excel(os.path.join(target_directory, 'PDGinfo.xlsx'), sheet_name=None)
                for sheet_name, sheet_df in target_df.items():
                    if sheet_df.empty or sheet_df.columns.size == 0:
                        continue
                c3d_input_related_target = get_target_from_file(input_csv_c3d_filename, target_df)
                if c3d_input_related_target is not None:
                    target_column = 'OFF - Hoehn & Yahr ' if 'off' in input_csv_c3d_filename else 'ON - Hoehn & Yahr ' if 'on' in input_csv_c3d_filename else None
                    if target_column and target_column in c3d_input_related_target.columns:
                        c3d_input_related_target_data = c3d_input_related_target[target_column].iloc[0]
                        c3d_input_related_target_series = pd.Series([c3d_input_related_target_data] * len(input_pelvic_c3d_data))
                        c3d_X = input_pelvic_c3d_data
                        c3d_y = c3d_input_related_target_series
                        c3d_X = c3d_X.reset_index(drop=True)
                        c3d_y = c3d_y.reset_index(drop=True)
                        if is_training_file(input_csv_c3d_filename):
                            if not c3d_X.empty or not c3d_X.isna().any().any():
                                c3d_train_all_X = pd.concat([c3d_train_all_X, c3d_X], ignore_index=True)
                            if not c3d_y.empty:
                                c3d_train_all_y = pd.concat([c3d_train_all_y, c3d_y], ignore_index=True)
                        else:
                            if not c3d_X.empty or not c3d_X.isna().any().any():
                                c3d_test_all_X = pd.concat([c3d_test_all_X, c3d_X], ignore_index=True)
                            if not c3d_y.empty:
                                c3d_test_all_y = pd.concat([c3d_test_all_y, c3d_y], ignore_index=True)
            except Exception as e:
                print(f"Error occurred while fetching target for file {input_csv_c3d_filename}: {e}")

    train_indices = c3d_train_all_X.index
    test_indices = c3d_test_all_X.index

    c3d_train_X_cleaned = c3d_train_all_X.dropna()
    c3d_test_X_cleaned = c3d_test_all_X.dropna()

    train_indices_after_dropna = c3d_train_X_cleaned.index
    test_indices_after_dropna = c3d_test_X_cleaned.index

    dropped_train_indices = train_indices.difference(train_indices_after_dropna)
    dropped_test_indices = test_indices.difference(test_indices_after_dropna)

    c3d_train_all_y = c3d_train_all_y.drop(dropped_train_indices)
    c3d_test_all_y = c3d_test_all_y.drop(dropped_test_indices)

    c3d_train_all_y = c3d_train_all_y.reset_index(drop=True)
    c3d_test_all_y = c3d_test_all_y.reset_index(drop=True)

    c3d_train_X = c3d_train_X_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)
    y_train_c3d = c3d_train_all_y.sample(frac=1, random_state=42).reset_index(drop=True)
    c3d_test_X = c3d_test_X_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)
    y_test_c3d = c3d_test_all_y.sample(frac=1, random_state=42).reset_index(drop=True)
    if c3d_train_X is None or len(c3d_train_X) == 0:
        print("Data is empty or None.0")
    else:
        print("Data is not empty.0")
    scaler = StandardScaler()
    X_train_scaled_c3d = scaler.fit_transform(c3d_train_X)
    X_test_scaled_c3d = scaler.transform(c3d_test_X)
    if X_train_scaled_c3d is None or len(X_train_scaled_c3d) == 0:
        print("Scaled Data is empty or None.1")
    else:
        print("Scaled Data is not empty.1")
    # Oversample X_train, y_train
    smote = SMOTE(random_state=42)
    X_train_scaled_c3d, y_train_c3d = smote.fit_resample(X_train_scaled_c3d, y_train_c3d)
    return X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d

def cm_displayer(cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.show()


def svm_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d):
    global optimized_classifiers

    def objective(trial):
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        C = trial.suggest_float('C', 0.1, 10.0, log=True)  # Using a log scale for C

        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 4)
            svc = SVC(kernel=kernel, C=C, degree=degree, random_state=42)
        elif kernel in ['rbf', 'sigmoid']:
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            svc = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
        else:
            svc = SVC(kernel=kernel, C=C, random_state=42)

        # StratifiedKFold cross-validation for evaluation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced number of splits
        scores = []
        for train_index, val_index in skf.split(X_train_scaled_c3d, y_train_c3d):
            X_train_fold, X_val_fold = X_train_scaled_c3d[train_index], X_train_scaled_c3d[val_index]
            y_train_fold, y_val_fold = y_train_c3d[train_index], y_train_c3d[val_index]

            svc.fit(X_train_fold, y_train_fold)
            y_pred = svc.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred, average='weighted')
            scores.append(score)

        # Report the mean of the scores to Optuna
        return np.mean(scores)

    # Create an Optuna study and optimize with minimal trials and pruning
    study = optuna.create_study(direction='maximize', study_name="SVM Optimization",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=10, n_jobs=12)  # Reduced number of trials

    # Retrieve the best parameters and model
    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    # Train the best model using the entire dataset
    svm_classifier_c3d = SVC(**best_params, random_state=42)
    svm_classifier_c3d.fit(X_train_scaled_c3d, y_train_c3d)
    optimized_classifiers['SVC'] = svm_classifier_c3d


def decision_tree_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d):
    global optimized_classifiers
    def objective(trial):
        max_features_option = trial.suggest_categorical('max_features_option', ['int', 'float', 'sqrt', 'log2', None])
        if max_features_option == 'int':
            max_features = trial.suggest_int('max_features_int', 1, X_train_scaled_c3d.shape[1])
        elif max_features_option == 'float':
            max_features = trial.suggest_float('max_features_float', 0.1, 1.0)
        else:
            max_features = max_features_option
        param = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
            'max_depth': trial.suggest_int('max_depth', 1, 32, log=True),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.5),
            'max_features': max_features,
            'random_state': 42,
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 50, log=True),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0, 0.2),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'ccp_alpha': trial.suggest_float('ccp_alpha', 0, 0.1)
        }

        clf = DecisionTreeClassifier(**param)
        score = cross_val_score(clf, X_train_scaled_c3d, y_train_c3d, n_jobs=12, cv=10, scoring='f1_weighted').mean()
        return score

    study = optuna.create_study(direction='maximize', study_name="Decision Tree Optimization")
    study.optimize(objective, n_trials=40, n_jobs=12)

    best_params = study.best_params
    print("Best parameters:", best_params)
    best_params.pop('max_features_option', None)  # Remove 'max_features_option' if present
    best_params.pop('max_features_int', None)  # Remove 'max_features_int' if present
    best_params.pop('max_features_float', None)  # Remove 'max_features_float' if present
    dt_classifier_c3d = DecisionTreeClassifier(**best_params)
    dt_classifier_c3d.fit(X_train_scaled_c3d, y_train_c3d)
    optimized_classifiers['DTC'] = dt_classifier_c3d

def knn_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d):
    global optimized_classifiers

    def objective(trial):
        # Suggest values for the hyperparameters
        n_neighbors = trial.suggest_int('n_neighbors', 2, 30)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        leaf_size = trial.suggest_int('leaf_size', 1, 50)
        p = trial.suggest_float('p', 1, 10)
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])

        # Create the KNN classifier with the suggested hyperparameters
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                                          leaf_size=leaf_size, p=p, metric=metric)

        # Perform cross-validation
        score = cross_val_score(classifier, X_train_scaled_c3d, y_train_c3d, n_jobs=12, cv=5,
                                scoring='f1_weighted').mean()
        return score

    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize', study_name="KNeighbors Optimization")
    study.optimize(objective, n_trials=15, n_jobs=12)

    best_params = study.best_params
    best_score = study.best_value
    print("Best parameters:", best_params)
    print("Best score:", best_score)

    # Train the classifier with the best parameters found
    knn_classifier_c3d = KNeighborsClassifier(**best_params)
    knn_classifier_c3d.fit(X_train_scaled_c3d, y_train_c3d)
    optimized_classifiers['KNNC'] = knn_classifier_c3d



def naive_bayes_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d):
    global optimized_classifiers
    # Define the objective function for Optuna
    def objective(trial):
        var_smoothing = trial.suggest_float('var_smoothing', 1e-9, 1e0, log=True)

        nb_classifier = GaussianNB(var_smoothing=var_smoothing)

        # StratifiedKFold cross-validation for evaluation
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
        scores = []
        for train_index, val_index in skf.split(X_train_scaled_c3d, y_train_c3d):
            X_train_fold, X_val_fold = X_train_scaled_c3d[train_index], X_train_scaled_c3d[val_index]
            y_train_fold, y_val_fold = y_train_c3d[train_index], y_train_c3d[val_index]

            nb_classifier.fit(X_train_fold, y_train_fold)
            y_pred = nb_classifier.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred, average='weighted')
            scores.append(score)

        return np.mean(scores)

    # Create an Optuna study and optimize
    study = optuna.create_study(direction='maximize', study_name="GaussianNB Optimization")
    study.optimize(objective, n_trials=30, n_jobs=12)

    # Retrieve the best parameters and model
    best_params = study.best_params
    print(f"Best parameters found: {best_params}")

    # Train the best model using the entire dataset
    nb_classifier = GaussianNB(**best_params)
    nb_classifier.fit(X_train_scaled_c3d, y_train_c3d)
    optimized_classifiers['GNB'] = nb_classifier

def bagging_classifier_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d):
    global optimized_classifiers

    # Define the objective function for Optuna
    def objective(trial):
        dt_classifier = DecisionTreeClassifier()

        # Suggest hyperparameters for BaggingClassifier
        n_estimators = trial.suggest_int('n_estimators', 20, 40)
        max_features = trial.suggest_float('max_features', 0.92, 1.0)
        max_samples = trial.suggest_float('max_samples', 0.1, 0.3)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        bootstrap_features = trial.suggest_categorical('bootstrap_features', [True, False])
        warm_start = False

        # Ensure oob_score is only enabled if bootstrap is True
        if bootstrap:
            oob_score = trial.suggest_categorical('oob_score', [True, False])
        else:
            oob_score = False


        bc_classifier = BaggingClassifier(
            estimator=dt_classifier,  # Use `estimator` instead of `base_estimator`
            n_estimators=n_estimators,
            max_features=max_features,
            max_samples=max_samples,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            warm_start=warm_start,
            oob_score=oob_score,
            n_jobs=12,
            random_state=1
        )

        # StratifiedKFold cross-validation for evaluation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)  # Adjust n_splits if needed
        scores = []
        for train_index, val_index in skf.split(X_train_scaled_c3d, y_train_c3d):
            X_train_fold, X_val_fold = X_train_scaled_c3d[train_index], X_train_scaled_c3d[val_index]
            y_train_fold, y_val_fold = y_train_c3d[train_index], y_train_c3d[val_index]

            bc_classifier.fit(X_train_fold, y_train_fold)
            y_pred = bc_classifier.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred, average='weighted')
            scores.append(score)

        return np.mean(scores)

    # Create an Optuna study and optimize
    study = optuna.create_study(direction='maximize', study_name="BaggingClassifier Optimization")
    study.optimize(objective, n_trials=20, n_jobs=12)

    # Retrieve the best parameters and model
    best_params = study.best_params
    print(f"Best parameters found: {best_params}")

    # Train the best model using the entire dataset
    bc_classifier = BaggingClassifier(
        estimator=DecisionTreeClassifier(),  # Use `estimator` instead of `base_estimator`
        **best_params,
        n_jobs=12,
        random_state=1
    )
    bc_classifier.fit(X_train_scaled_c3d, y_train_c3d)
    optimized_classifiers['BagC'] = bc_classifier


def ada_boost_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d):
    global optimized_classifiers
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 70, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.08, 0.3)
        algorithm = trial.suggest_categorical('algorithm', ['SAMME'])

        abc_classifier = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=1
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        scores = []
        for train_index, val_index in skf.split(X_train_scaled_c3d, y_train_c3d):
            X_train_fold, X_val_fold = X_train_scaled_c3d[train_index], X_train_scaled_c3d[val_index]
            y_train_fold, y_val_fold = y_train_c3d[train_index], y_train_c3d[val_index]

            abc_classifier.fit(X_train_fold, y_train_fold)
            y_pred = abc_classifier.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred, average='weighted')
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction='maximize', study_name="AdaBoostClassifier Optimization")
    study.optimize(objective, n_trials=20, n_jobs=12)

    best_params = study.best_params
    print(f"Best parameters found: {best_params}")

    abc_classifier = AdaBoostClassifier(
        **best_params,
        random_state=1
    )
    abc_classifier.fit(X_train_scaled_c3d, y_train_c3d)
    optimized_classifiers['ABC'] = abc_classifier


def catboost_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d):
    global optimized_classifiers
    # Define the objective function for Optuna
    def objective(trial):
        # Suggest hyperparameters for CatBoostClassifier
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'n_estimators': trial.suggest_int('n_estimators', 10, 100),
            'random_state': 1,
            'verbose': False,
            'thread_count': 1,
        }

        catboost_classifier = CatBoostClassifier(**param)

        # StratifiedKFold cross-validation for evaluation
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        scores = []
        for train_index, val_index in skf.split(X_train_scaled_c3d, y_train_c3d):
            X_train_fold, X_val_fold = X_train_scaled_c3d[train_index], X_train_scaled_c3d[val_index]
            y_train_fold, y_val_fold = y_train_c3d[train_index], y_train_c3d[val_index]

            try:
                catboost_classifier.fit(X_train_fold, y_train_fold)
            except Exception as e:
                print(f"An error occurred: {e}")
                return 0.0
            y_pred = catboost_classifier.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred, average='weighted')
            scores.append(score)

        return np.mean(scores)

    # Create an Optuna study and optimize
    study = optuna.create_study(direction='maximize', study_name="CatBoostClassifier Optimization")
    study.optimize(objective, n_trials=15, n_jobs=-1)

    # Retrieve the best parameters and model
    best_params = study.best_params
    print(f"Best parameters found: {best_params}")

    # Train the best model using the entire dataset
    catboost_classifier = CatBoostClassifier(**best_params, random_state=1, verbose=False)
    catboost_classifier.fit(X_train_scaled_c3d, y_train_c3d)
    optimized_classifiers['CBC'] = catboost_classifier


def lgbm_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d):
    global optimized_classifiers

    def objective(trial):
        param = {
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
            'num_leaves': trial.suggest_int('num_leaves', 20, 50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),  # Increased the upper limit
            'subsample_for_bin': trial.suggest_int('subsample_for_bin', 200000, 300000),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.1),  # Reduced range
            'min_child_weight': trial.suggest_float('min_child_weight', 0.001, 0.1),  # Reduced range
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),  # Increased flexibility
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 0.5),
            'verbose': -1
        }

        lgbm_classifier = LGBMClassifier(**param, random_state=1)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
        scores = []
        for train_index, val_index in skf.split(X_train_scaled_c3d, y_train_c3d):
            X_train_fold, X_val_fold = X_train_scaled_c3d[train_index], X_train_scaled_c3d[val_index]
            y_train_fold, y_val_fold = y_train_c3d[train_index], y_train_c3d[val_index]

            # Fit the model with early stopping
            lgbm_classifier.fit(
                X_train_fold,
                y_train_fold
            )

            y_pred = lgbm_classifier.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred, average='weighted')
            scores.append(score)

        return np.mean(scores)

    study = optuna.create_study(direction='maximize', study_name="LGBMClassifier Optimization")
    study.optimize(objective, n_trials=10, n_jobs=12)

    best_params = study.best_params
    print(f"Best parameters found: {best_params}")

    lgbm_classifier = LGBMClassifier(**best_params, random_state=1)
    lgbm_classifier.fit(X_train_scaled_c3d, y_train_c3d)
    optimized_classifiers['LGBMC'] = lgbm_classifier

def xgboost_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d):
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Fit and transform y to encode labels
    y_encoded = label_encoder.fit_transform(y_train_c3d)
    global optimized_classifiers
    # Define the objective function for Optuna
    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 10, 300),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_encoded)),
            'tree_method': 'hist',
            'eval_metric': 'mlogloss'
        }

        xgb_clf = XGBClassifier(**param)

        # StratifiedKFold cross-validation for evaluation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_index, val_index in skf.split(X_train_scaled_c3d, y_encoded):
            X_train_fold, X_val_fold = X_train_scaled_c3d[train_index], X_train_scaled_c3d[val_index]
            y_train_fold, y_val_fold = y_encoded[train_index], y_encoded[val_index]

            xgb_clf.fit(X_train_fold, y_train_fold)
            y_pred = xgb_clf.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred, average='weighted')
            scores.append(score)

        return np.mean(scores)

    # Create an Optuna study and optimize
    study = optuna.create_study(direction='maximize', study_name="XGBoost Optimization")
    study.optimize(objective, n_trials=10, n_jobs=12)

    # Retrieve the best parameters and model
    best_params = study.best_params
    print(f"Best parameters found: {best_params}")

    # Train the best model using the entire dataset
    xgb_clf = XGBClassifier(**best_params, random_state=42)
    xgb_clf.fit(X_train_scaled_c3d, y_encoded)
    optimized_classifiers['XGBC'] = xgb_clf

def random_forest_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d):
    global optimized_classifiers

    # Define the objective function for Optuna
    def objective(trial):
        max_features_option = trial.suggest_categorical('max_features_option', ['int', 'float', 'sqrt', 'log2', None])
        if max_features_option == 'int':
            max_features = trial.suggest_int('max_features_int', 1, X_train_scaled_c3d.shape[1])
        elif max_features_option == 'float':
            max_features = trial.suggest_float('max_features_float', 0.1, 1.0)
        else:
            max_features = max_features_option
        param = {
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'max_depth': trial.suggest_categorical('max_depth', [10, 20, 30, 40, None]),
            'max_features': max_features,
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'n_estimators': trial.suggest_int('n_estimators', 10, 30),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
        }

        rf_classifier = RandomForestClassifier(**param, random_state=1)

        # StratifiedKFold cross-validation for evaluation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        scores = []
        for train_index, val_index in skf.split(X_train_scaled_c3d, y_train_c3d):
            X_train_fold, X_val_fold = X_train_scaled_c3d[train_index], X_train_scaled_c3d[val_index]
            y_train_fold, y_val_fold = y_train_c3d[train_index], y_train_c3d[val_index]

            rf_classifier.fit(X_train_fold, y_train_fold)
            y_pred = rf_classifier.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred, average='weighted')
            scores.append(score)

        return np.mean(scores)

    # Create an Optuna study and optimize
    study = optuna.create_study(direction='maximize', study_name="RandomForestClassifier Optimization")
    study.optimize(objective, n_trials=10, n_jobs=12)

    # Retrieve the best parameters and model
    best_params = study.best_params

    best_params.pop('max_features_option', None)  # Remove 'max_features_option' if present
    best_params.pop('max_features_int', None)  # Remove 'max_features_int' if present
    best_params.pop('max_features_float', None)  # Remove 'max_features_float' if present
    print(f"Best parameters found: {best_params}")

    # Train the best model using the entire dataset
    rf_classifier = RandomForestClassifier(**best_params, random_state=1)
    rf_classifier.fit(X_train_scaled_c3d, y_train_c3d)
    optimized_classifiers['RFC'] = rf_classifier

def metric_printer(classifier_name, classifier, X_test_scaled, y_test):
    y_pred = classifier.predict(X_test_scaled)
    if classifier_name == 'XGBClassifier':
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)
        y_pred = label_encoder.inverse_transform(y_pred)

    print("Classifier: ", classifier_name)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    print("Precision: ", precision)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    print("Recall: ", recall)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1 Score: ", f1)
    print('____________________________________')

    c3d_results[classifier_name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


if __name__ == "__main__":
    X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d = load_data()

    svm_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d)
    decision_tree_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d)
    knn_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d)
    naive_bayes_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d)
    random_forest_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d)
    bagging_classifier_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d)
    ada_boost_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d)
    catboost_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d)
    lgbm_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d)
    xgboost_optimization(X_train_scaled_c3d, X_test_scaled_c3d, y_train_c3d, y_test_c3d)
    c3d_results = {}
    # Train and evaluate each classifier
    for name,clf in optimized_classifiers.items():
        metric_printer(name,clf,X_test_scaled_c3d, y_test_c3d)

    



