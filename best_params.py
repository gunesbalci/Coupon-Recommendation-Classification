import lightgbm
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, log_loss
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_LGBM_trial_Model(trial):
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
        'verbosity': -1,
        'n_jobs': -1
    }
    return LGBMClassifier(**params)

def get_bagging_trial_Model(trial):
    max_depth = trial.suggest_int('max_depth', 3, 20)
    base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    params = {
        'estimator': base_tree,
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_samples': trial.suggest_float('max_samples', 0.6, 1.0),
        'max_features': trial.suggest_float('max_features', 0.6, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1  # Tüm işlemci çekirdeklerini kullanarak hızlı eğitir
    }
    return BaggingClassifier(**params)

def get_SVM_trial_Model(trial):
    C = trial.suggest_float('C', 0.01, 100.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
    gamma = 'scale'
    if kernel == 'rbf':
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            C=C, 
            kernel=kernel, 
            gamma=gamma, 
            probability=True,  
            random_state=42
        ))
    ])

def objective_with_metrics(trial, X, y, model_type='lgbm'):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    roc_auc_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    log_losses = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_type == 'lgbm':
            model = get_LGBM_trial_Model(trial)
        elif model_type == 'bagging':
            model = get_bagging_trial_Model(trial)
        elif model_type == 'svm':
            model = get_SVM_trial_Model(trial)

        if model_type == 'lgbm':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lightgbm.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            model.fit(X_train, y_train)
        
        # Olasılıklar ve sınıf tahminleri (eşik değeri varsayılan 0.5)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        # Metriklerin hesaplanması
        roc_auc_scores.append(roc_auc_score(y_val, y_pred_proba))
        f1_scores.append(f1_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_val, y_pred, zero_division=0))
        log_losses.append(log_loss(y_val, y_pred_proba))

    trial.set_user_attr("f1_score", np.mean(f1_scores))
    trial.set_user_attr("log_loss", np.mean(log_losses))
    trial.set_user_attr("precision", np.mean(precision_scores))
    trial.set_user_attr("recall", np.mean(recall_scores))

    return np.mean(roc_auc_scores)

def learn_best_params(x_train, y_train, model_type='lgbm', n_trials=20):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_with_metrics(trial, x_train, y_train, model_type), n_trials=n_trials)

    print(f"En iyi parametreler: {study.best_params}")
    print(f"En iyi AUC Skoru: {study.best_value:.4f}")

    best_f1_trial = max(study.trials, key=lambda t: t.user_attrs.get("f1_score", 0))
    print(f"En iyi F1 Skoru: {best_f1_trial.user_attrs['f1_score']:.4f}")
    print(f"Bu F1'i veren Parametreler: {best_f1_trial.params}")

    best_logloss_trial = min(study.trials, key=lambda t: t.user_attrs.get("log_loss", float('inf')))
    print(f"En düşük Log Loss Değeri: {best_logloss_trial.user_attrs['log_loss']:.4f}")
    print(f"Bu Log Loss'u veren Parametreler: {best_logloss_trial.params}")

    best_precision_trial = max(study.trials, key=lambda t: t.user_attrs.get("precision", 0))
    print(f"En iyi Precision Skoru: {best_precision_trial.user_attrs['precision']:.4f}")
    print(f"Bu Precision'ı veren Parametreler: {best_precision_trial.params}")

    best_recall_trial = max(study.trials, key=lambda t: t.user_attrs.get("recall", 0))
    print(f"En iyi Recall Skoru: {best_recall_trial.user_attrs['recall']:.4f}")
    print(f"Bu Recall'u veren Parametreler: {best_recall_trial.params}")

    file_name = f"Results/Test/BestParams/{model_type}/{model_type}_log.txt"
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(f"Model: {model_type}\n")
        f.write(f"Best AUC Score: {study.best_value:.4f}\n")
        f.write(f"Parameters for Best AUC: {study.best_params}\n\n")
        f.write(f"Best F1 Score: {best_f1_trial.user_attrs['f1_score']:.4f}\n")
        f.write(f"Parameters for Best F1: {best_f1_trial.params}\n\n")
        f.write(f"Best Log Loss Value: {best_logloss_trial.user_attrs['log_loss']:.4f}\n")
        f.write(f"Parameters for Best Log Loss: {best_logloss_trial.params}\n\n")
        f.write(f"Best Precision Score: {best_precision_trial.user_attrs['precision']:.4f}\n")
        f.write(f"Parameters for Best Precision: {best_precision_trial.params}\n\n")
        f.write(f"Best Recall Score: {best_recall_trial.user_attrs['recall']:.4f}\n")
        f.write(f"Parameters for Best Recall: {best_recall_trial.params}\n")
        f.write("-" * 80 + "\n")