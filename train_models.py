from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import numpy as np
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_auc_score, f1_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split
from comparing import extract_data
import warnings
warnings.filterwarnings("ignore")

def get_best_model(model_type, best_params):
    if model_type == 'lgbm':
        model = LGBMClassifier(**best_params, random_state=42, objective='binary', verbosity=-1, n_jobs=-1)
    elif model_type == 'bagging':
        base_tree = DecisionTreeClassifier(max_depth=best_params.get('max_depth', 10), random_state=42)
        best_params = {k: v for k, v in best_params.items() if k != 'max_depth'}
        model = BaggingClassifier(estimator=base_tree, **best_params, random_state=42, n_jobs=-1)
    elif model_type == 'svm':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(C=best_params.get('C', 1.0), kernel=best_params.get('kernel', 'rbf'), 
                        gamma=best_params.get('gamma', 'scale'), probability=True, random_state=42))
        ])
    elif model_type == 'stacking':
        estimators = [
            ('lgb', get_best_model("lgbm", lgbm_best_logloss_params)),
            ('bagging', get_best_model("bagging", bagging_best_logloss_params)),
            ('svm', get_best_model("svm", svm_best_logloss_params))
        ]
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=best_params.get('meta_C', 1.0), random_state=42),
            cv=5,
            passthrough=best_params.get('passthrough', False),
            n_jobs=-1
        )
    return model

def evaluate_with_repeated_splits(best_params, model_type='lgbm', n_repeats=5, best_metric="ROC-AUC"):
    test_roc_scores = []
    test_f1_scores = []
    test_precision_scores = []
    test_recall_scores = []
    test_logloss_scores = []
    
    for i in range(n_repeats):
        current_seed = i * 42  
        X_train, X_test, y_train, y_test = extract_data(apply_target_encode=True, random_state=current_seed)
        
        model = get_best_model(model_type, best_params)    
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        test_roc_scores.append(roc_auc_score(y_test, y_pred_proba))
        test_f1_scores.append(f1_score(y_test, y_pred))
        test_precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
        test_recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
        test_logloss_scores.append(log_loss(y_test, y_pred_proba))
        
    mean_roc = np.mean(test_roc_scores)
    std_roc = np.std(test_roc_scores)
    mean_f1 = np.mean(test_f1_scores)
    mean_logloss = np.mean(log_loss_scores := test_logloss_scores) # log_loss_scores
    mean_precision = np.mean(test_precision_scores)
    mean_recall = np.mean(test_recall_scores)
    
    print("\n=== REPEATED FINAL TEST PERFORMANCE (Mean ± Std) ===")
    print(f"Test ROC-AUC : {mean_roc:.4f} (±{std_roc:.4f})")
    print(f"Test F1      : {mean_f1:.4f} (±{np.std(test_f1_scores):.4f})")
    print(f"Test Log Loss: {mean_logloss:.4f} (±{np.std(test_logloss_scores):.4f})")
    print(f"Test Precision: {mean_precision:.4f} (±{np.std(test_precision_scores):.4f})")
    print(f"Test Recall   : {mean_recall:.4f} (±{np.std(test_recall_scores):.4f})")
    
    # Sonuçları dosyaya kaydetme
    file_name = f"Results/Test/BestParams/{model_type}/{model_type}_{best_metric}.txt"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write("\n=== REPEATED FINAL TEST PERFORMANCE (5 Splits) ===\n")
        f.write(f"Mean Test ROC-AUC: {mean_roc:.4f} (±{std_roc:.4f})\n")
        f.write(f"Mean Test F1: {mean_f1:.4f} (±{np.std(test_f1_scores):.4f})\n")
        f.write(f"Mean Test Log Loss: {mean_logloss:.4f} (±{np.std(test_logloss_scores):.4f})\n")
        f.write(f"Mean Test Precision: {mean_precision:.4f} (±{np.std(test_precision_scores):.4f})\n")
        f.write(f"Mean Test Recall: {mean_recall:.4f} (±{np.std(test_recall_scores):.4f})\n")
        f.write("=" * 80 + "\n")    

lgbm_best_roc_auc_params = {
    'boosting_type': 'gbdt', 'n_estimators': 544,
    'learning_rate': 0.07782116242300102, 'max_depth': 9, 'num_leaves': 132,
    'min_child_samples': 85, 'subsample': 0.6547137593717837,
    'colsample_bytree': 0.7194299908706323}
lgbm_best_f1_params = {
    'boosting_type': 'gbdt', 'n_estimators': 666, 
    'learning_rate': 0.125050755654907, 'max_depth': 9, 'num_leaves': 39, 
    'min_child_samples': 31, 'subsample': 0.8153832785683361, 
    'colsample_bytree': 0.620631496549414}
lgbm_best_logloss_params = {
    'boosting_type': 'gbdt', 'n_estimators': 740, 
    'learning_rate': 0.06306019877700901, 'max_depth': 10, 'num_leaves': 38, 
    'min_child_samples': 31, 'subsample': 0.7874669211557248, 
    'colsample_bytree': 0.6360101706105831}
lgbm_best_precision_params = {
    'boosting_type': 'gbdt', 'n_estimators': 780, 
    'learning_rate': 0.09076789709866397, 'max_depth': 10, 'num_leaves': 51, 
    'min_child_samples': 78, 'subsample': 0.9154236718908, 
    'colsample_bytree': 0.6252858889480076
}
lgbm_best_recall_params = {
    'boosting_type': 'gbdt', 'n_estimators': 122, 
    'learning_rate': 0.010223194657237141, 'max_depth': 8, 'num_leaves': 62, 
    'min_child_samples': 38, 'subsample': 0.7350610183598968, 
    'colsample_bytree': 0.921260062183799}

bagging_best_roc_auc_params = {'max_depth': 18, 'n_estimators': 168, 
    'max_samples': 0.6017235035272963, 'max_features': 0.7742768390124075, 
    'bootstrap': True}
bagging_best_f1_params = {'max_depth': 14, 'n_estimators': 144, 
    'max_samples': 0.7356578372791587, 'max_features': 0.7574599126927946, 
    'bootstrap': True}
bagging_best_logloss_params = {'max_depth': 15, 'n_estimators': 221, 
    'max_samples': 0.7027843113840364, 'max_features': 0.8780643032270454, 
    'bootstrap': True}
bagging_best_precision_params = {'max_depth': 19, 'n_estimators': 181, 
    'max_samples': 0.9944826843141412, 'max_features': 0.9488698798342647, 
    'bootstrap': True}
bagging_best_recall_params = {'max_depth': 4, 'n_estimators': 111, 
    'max_samples': 0.9460569621099121, 'max_features': 0.6682658931833065, 
    'bootstrap': True}

svm_best_roc_auc_params = {'C': 1.7234532170464567, 'kernel': 'rbf', 'gamma': 'scale'}
svm_best_f1_params = {'C': 1.7234532170464567, 'kernel': 'rbf', 'gamma': 'scale'}
svm_best_logloss_params = {'C': 1.7234532170464567, 'kernel': 'rbf', 'gamma': 'scale'}
svm_best_precision_params = {'C': 4.936332527260067, 'kernel': 'rbf', 'gamma': 'scale'}
svm_best_recall_params = {'C': 0.021519260344323853, 'kernel': 'rbf', 'gamma': 'auto'}

stacking_best_roc_auc_params = {'meta_C': 1.2934508669159068, 'passthrough': False}
stacking_best_f1_params = {'meta_C': 0.002906966654561084, 'passthrough': False}
stacking_best_logloss_params = {'meta_C': 7.536579499191063, 'passthrough': True}
stacking_best_precision_params = {'meta_C': 7.536579499191063, 'passthrough': True}
stacking_best_recall_params = {'meta_C': 0.002906966654561084, 'passthrough': False}

def train_model(data, best_params, model_type='lgbm', best_metric="ROC-AUC", save=False): 
    X_train, X_test, y_train, y_test = data
    
    model = get_best_model(model_type, best_params)    
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    logloss_score = log_loss(y_test, y_pred_proba)

    if save:
        model_file_name = f"Results/Test/BestParams/{model_type}/best_model_{best_metric}.pkl"
        with open(model_file_name, "wb") as model_file:
            pickle.dump(model, model_file)

    print("\n=== FINAL TEST PERFORMANCE ===")
    print(f"Test ROC-AUC : {roc_auc:.4f}")
    print(f"Test F1      : {f1:.4f}")
    print(f"Test Log Loss: {logloss_score:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall   : {test_recall:.4f}")