from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, HistGradientBoostingClassifier)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from dataset_encode import encode_train_test_datasets
from dataset_preprocess import create_train_test_datasets
import matplotlib.pyplot as plt
import pandas as pd

# EXTRACTING TRAIN AND TEST DATA FROM DATASET
def extract_data(apply_target_encode=True):
    df = pd.read_csv("Dataset/in-vehicle-coupon-recommendation.csv")
    X_train, X_test, y_train, y_test = create_train_test_datasets(df)
    X_train, X_test = encode_train_test_datasets(X_train, X_test, y_train, apply_target_encode)
    return X_train, X_test, y_train, y_test

def calc_visualize_result(y_test, y_pred, y_pred_proba, model_name, print_on):
    acc = accuracy_score(y_test, y_pred)
    c_report = classification_report(y_test, y_pred)
    c_matrix = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    if print_on:
        print("Accuracy:", acc)
        print("\nDetaylı Sınıflandırma Raporu:\n", c_report)
        print("Confusion Matrix:\n", c_matrix)
        print(f"ROC-AUC Skoru: {auc_score:.4f}")

        txt_filename = f"Results/Test/{model_name}_results.txt"
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(f"Model Adı: {model_name}\n")
            f.write("=" * 40 + "\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"ROC-AUC Skoru: {auc_score:.4f}\n\n")
            f.write(f"Confusion Matrix: \n{c_matrix}\n\n")
            f.write("Classification Report:\n")
            f.write(c_report)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr,
        tpr,
        color="blue",
        label=f"ROC Curve (AUC = {auc_score:.2f})",
        lw=2,
    )
    plt.plot(
        [0, 1], [0, 1], color="red", linestyle="--"
    )  # Rastgele tahmin çizgisi
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Eğrisi")
    plt.legend(loc="lower right")
    plt.grid(True)
    if print_on: 
        plt.show()
        img_filename = f"Results/Test/{model_name}_roc_curve.png"
        plt.savefig(img_filename, dpi=300, bbox_inches="tight")
    plt.close()

    return auc_score

def train_LogisticReg(X_train, X_test, y_train, y_test, print_on):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return calc_visualize_result(y_test, y_pred, y_pred_proba,"Logistic Regression", print_on)

def train_RandomForest(X_train, X_test, y_train, y_test, print_on):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    return calc_visualize_result(y_test,y_pred,y_pred_proba,"RandomForest", print_on)

def train_XGBoost(X_train, X_test, y_train, y_test, print_on):
    xgb_model = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    return calc_visualize_result(y_test,y_pred,y_pred_proba,"XGBoost", print_on)

def train_AdaBoost(X_train, X_test, y_train, y_test, print_on):
    model = AdaBoostClassifier(
        n_estimators=100,learning_rate=1.0,random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return calc_visualize_result(y_test,y_pred,y_pred_proba,"XAdaBoost", print_on)

def train(model_name, X_train, X_test, y_train, y_test, print_on):

    model = LogisticRegression(max_iter=1000, random_state=42)
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42) 
    elif model_name == "XAdaBoost":
        model = AdaBoostClassifier(
            n_estimators=100,learning_rate=1.0,random_state=42)   
    elif model_name == "LightGBM":
        model = LGBMClassifier(
            n_estimators=100,learning_rate=1.0,random_state=42,objective="binary")
    elif model_name == "CatBoost":
        model = CatBoostClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42, verbose=False)
    elif model_name == "ExtraTrees":
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return calc_visualize_result(y_test,y_pred,y_pred_proba,model_name,print_on)