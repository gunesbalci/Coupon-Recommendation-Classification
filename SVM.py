import optuna
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

def objective_svm(trial,data):
    X_train, X_test, y_train, y_test = data
    # 1. Hiperparametre önerileri
    C = trial.suggest_float('C', 0.01, 100.0, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
    
    # Gamma parametresi sadece 'rbf' kernel seçildiğinde anlamlıdır
    gamma = 'scale'
    if kernel == 'rbf':
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    
    # 2. Pipeline: Önce ölçeklendirme (SVM için kritik), ardından SVC
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            C=C, 
            kernel=kernel, 
            gamma=gamma, 
            probability=True,  # Stacking için olasılık üretmesi şart
            random_state=42
        ))
    ])
    
    # 3. Eğitim ve Tahmin
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, preds)
    
    return score

def learn_best_params(data):
    # Optuna Çalışması
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_svm(trial, data), n_trials=20)

    print("En İyi SVM Skoru:", study.best_value)
    print("En İyi SVM Parametreleri:", study.best_params)

def get_best_svm_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            C=2.0016316871226083, 
            kernel='rbf', 
            gamma='auto', 
            probability=True,  
            random_state=42
        ))
    ])