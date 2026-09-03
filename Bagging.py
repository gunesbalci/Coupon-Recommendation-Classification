import optuna
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

def objective_bagging(trial,data):
    X_train, X_test, y_train, y_test = data
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
    
    model = BaggingClassifier(**params)
    model.fit(X_train, y_train)
    
    preds = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, preds)
    
    return score

def learn_best_params(data):
    # Optuna çalışması
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_bagging(trial, data), n_trials=30)

    print("En İyi Skor:", study.best_value)
    print("En İyi Parametreler:", study.best_params)

def get_best_bagging_model():
    base_tree = DecisionTreeClassifier(
        max_depth=18, 
        random_state=42
    )
    
    return BaggingClassifier(
        estimator=base_tree,  
        n_estimators=181,
        max_samples=0.932046941974057, 
        max_features=0.6490147364580608,
        bootstrap=True, 
        random_state=42,
        n_jobs=-1
    )