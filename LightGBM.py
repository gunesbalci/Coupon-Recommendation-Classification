from lightgbm import LGBMClassifier
import lightgbm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from dataset_encode import encode_train_test_datasets
from dataset_preprocess import create_train_test_datasets
from comparing import calc_visualize_result
from sklearn.model_selection import StratifiedKFold

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
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
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

def compare_results(results):
    df_results = pd.DataFrame(list(results.items()), columns=["Data Variety", "ROC-AUC Score"])

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(
        data=df_results, 
        x="ROC-AUC Score", 
        y="Data Variety", 
        hue="Data Variety",
        palette="viridis",
        legend=False
    )

    for p in ax.patches:
        width = p.get_width()
        if width > 0:  # Boş barları atlamak için
            ax.annotate(
                f'{width:.4f}',
                (width, p.get_y() + p.get_height() / 2.),  # Barın tam sağ ucu
                ha='left', va='center',                    # Yazıyı sola dayalı yap
                xytext=(5, 0),                             # Bardan hafif sağa kaydır
                textcoords='offset points',
                color='black', fontweight='bold', fontsize=11
            )

    plt.title("LightGBM Performance Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("ROC-AUC Score", fontsize=12)
    plt.ylabel("Data Variety", fontsize=12)
    min_score = df_results["ROC-AUC Score"].min()
    max_score = df_results["ROC-AUC Score"].max()
    plt.xlim(min_score - 0.01, max_score)
    plt.tight_layout()
    plt.show()

def Kfold_CV(data):
    X_train, X_test, y_train, y_test = data
    X_full = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

    # 2. 5-Fold Cross-Validation kurulumu
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        # Fold'lara ait train ve validation parçaları
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        
        # Modeli kurup eğitelim (varsayılan parametrelerinle)
        model = LGBMClassifier(random_state=42, verbose=-1)
        model.fit(X_tr, y_tr)
        
        # Olasılık tahmini alıp AUC hesaplayalım
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)
        cv_scores.append(score)

    # 3. Sonuçları yazdıralım
    print(f"Her Fold'un AUC Skorları: {[round(s, 4) for s in cv_scores]}")
    print(f"Ortalama CV AUC Skoru: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

def get_best_model():
    return LGBMClassifier(boosting_type='gbdt', n_estimators=582,
        learning_rate=0.09135301766719553, max_depth=6, num_leaves=110,
        min_child_samples=21, subsample=0.9841372878332021,
        colsample_bytree=0.6359476359169153, verbosity=-1, random_state=42, objective='binary')