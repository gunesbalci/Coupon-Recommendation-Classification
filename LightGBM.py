from lightgbm import LGBMClassifier
import lightgbm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import optuna
from sklearn.metrics import roc_auc_score
from dataset_encode import encode_train_test_datasets
from dataset_preprocess import create_train_test_datasets
from comparing import calc_visualize_result

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# EXTRACTING TRAIN AND TEST DATA FROM DATASET
def extract_data(apply_target_encoding=True):
    df = pd.read_csv("Dataset/in-vehicle-coupon-recommendation.csv")
    X_train, X_test, y_train, y_test = create_train_test_datasets(df)
    X_train, X_test = encode_train_test_datasets(X_train, X_test, y_train, apply_target_encoding)
    return X_train, X_test, y_train, y_test

def objective(trial, data):
    X_train, X_test, y_train, y_test = data
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'objective': 'binary',
        'verbosity': -1
    }
    
    model = LGBMClassifier(**params)
    
    # Early stopping ile eğitim
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lightgbm.early_stopping(stopping_rounds=50, verbose=False)]
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_pred_proba)
    
    return score

def learn_best_params(data):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, data), n_trials=50)

    print(f"En iyi parametreler: {study.best_params}")
    print(f"En iyi AUC Skoru: {study.best_value:.4f}")

def train(data, print_on):
    X_train, X_test, y_train, y_test = data
    model = LGBMClassifier(
        n_estimators=100,learning_rate=0.1,random_state=42,objective="binary",verbosity=-1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return calc_visualize_result(y_test,y_pred,y_pred_proba,"LightGBM",print_on)