import optuna
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from LightGBM import get_best_model
from Bagging import get_best_bagging_model
from SVM import get_best_svm_model

def objective_stacking(trial,data):
    X_train, X_test, y_train, y_test = data
    meta_C = trial.suggest_float('meta_C', 0.001, 10.0, log=True)
    passthrough = trial.suggest_categorical('passthrough', [True, False])
    
    estimators = [
        ('lgb', get_best_model()),
        ('bagging', get_best_bagging_model()),
        ('svm', get_best_svm_model())
    ]
    
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=meta_C, random_state=42),
        cv=5,
        passthrough=passthrough,
        n_jobs=-1
    )

    stacking_model.fit(X_train, y_train)
    preds = stacking_model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, preds)
    
    return score

def learn_best_params(data):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_stacking(trial, data), n_trials=20)

    print("En İyi Stacking Skoru:", study.best_value)
    print("En İyi Stacking Parametreleri:", study.best_params)