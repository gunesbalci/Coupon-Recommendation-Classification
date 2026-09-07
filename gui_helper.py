import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import (
    roc_auc_score, f1_score, log_loss, precision_score, recall_score,
    confusion_matrix, roc_curve
)
import pickle
from comparing import extract_data

df = pd.read_csv("Dataset/in-vehicle-coupon-recommendation.csv")

def heatmap_comparison():
    models = [
        "Logistic Regression",
        "Random Forest Classifier",
        "Support Vector Machine",
        "XGBoost Classifier",
        "AdaBoost Classifier",
        "LightGBM Classifier",
        "CatBoost Classifier",
        "ExtraTrees Classifier",
        "HistGradientBoosting Classifier",
        "Bagging Classifier",
        "Stacking Classifier",
        "Voting Classifier"
    ]

    results_data = {
        "Plain": [0.742, 0.786, 0.778, 0.815, 0.748, 0.815, 0.804, 0.749, 0.807, 0.808, 0.824, 0.816],
        "Time Rmv": [0.741, 0.787, 0.776, 0.813, 0.748, 0.812, 0.801, 0.751, 0.802, 0.808, 0.822, 0.813],
        "Gender Rmv": [0.741, 0.779, 0.774, 0.814, 0.748, 0.808, 0.799, 0.744, 0.806, 0.801, 0.817, 0.813],
        "Direction Rmv": [0.742, 0.791, 0.780, 0.817, 0.748, 0.818, 0.799, 0.753, 0.807, 0.809, 0.827, 0.818],
        "Target Enc Rmv": [0.742, 0.797, 0.786, 0.811, 0.747, 0.810, 0.801, 0.772, 0.808, 0.811, 0.822, 0.813],
    }

    df_scores = pd.DataFrame(results_data, index=models)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        df_scores,
        annot=True,  
        fmt=".3f", 
        cmap="RdYlGn", 
        linewidths=0.5,  
        cbar=True,
        ax=ax  # Eksen bağlantısı için şart
    )  

    ax.set_title("Heat Maps for Test ROC-AUC Score", fontsize=14)
    ax.set_xlabel("Dataset Varieties", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    fig.tight_layout()
    return fig

def plot_manual_results(model_name):
    auc, f1, log_loss, precision, recall = [], [], [], [], []
    if model_name == "lgbm":
        auc = [0.8280, 0.8272, 0.8296, 0.8281, 0.7821]
        f1 = [0.7928, 0.7951, 0.7970, 0.7937, 0.7742]
        log_loss = [0.5143, 0.5321, 0.5054, 0.5276, 0.5718]
        precision = [0.7739, 0.7745, 0.7708, 0.7767, 0.7765]
        recall = [0.8129, 0.8169, 0.8250, 0.8116, 0.8421]
    elif model_name == "bagging":
        auc = [0.8053, 0.8047, 0.8047, 0.8022, 0.7913]
        f1 = [0.7834, 0.7834, 0.7844, 0.7837, 0.7779]
        log_loss = [0.5340, 0.5351, 0.5313, 0.5321, 0.5529]
        precision = [0.7393, 0.7383, 0.7397, 0.7398, 0.7279]
        recall = [0.8333, 0.8344, 0.8349, 0.8334, 0.8355]
    elif model_name == "svm":
        auc = [0.7823, 0.7823, 0.7823, 0.7812, 0.7490]
        f1 = [0.7674, 0.7674, 0.7674, 0.7675, 0.7540]
        log_loss = [0.5555, 0.5555, 0.5555, 0.5559, 0.5861]
        precision = [0.7324, 0.7324, 0.7324, 0.7365, 0.7052]
        recall = [0.8061, 0.8061, 0.8061, 0.8013, 0.8102]
    elif model_name == "stacking":
        auc = [0.8308, 0.8278, 0.8299, 0.8299, 0.8278]
        f1 = [0.7983, 0.8008, 0.7983, 0.7985, 0.8007]
        log_loss = [0.5002, 0.5464, 0.5005, 0.5005, 0.5464]
        precision = [0.7660, 0.7440, 0.7680, 0.7684, 0.7439]
        recall = [0.8334, 0.8671, 0.8312, 0.8312, 0.8669]
        
    data = {
        'Optimization Goal': [
            'Optimized for AUC', 
            'Optimized for F1', 
            'Optimized for Log Loss',
            'Optimized for Precision', 
            'Optimized for Recall'
        ],
        'ROC-AUC':   auc,
        'F1-Score':  f1,
        'Log Loss':  log_loss,
        'Precision': precision,
        'Recall':    recall
    }

    df_results = pd.DataFrame(data)
    metrics_names = ['ROC-AUC', 'F1-Score', 'Log Loss', 'Precision', 'Recall']
    df_melted = df_results.melt(id_vars='Optimization Goal', value_vars=metrics_names, 
                                var_name='Metric', value_name='Score')
    
    fig = px.bar(
        df_melted, 
        x='Metric', 
        y='Score', 
        color='Optimization Goal', 
        barmode='group',
        title=f"{model_name} - Optimizasyon Stratejilerinin Karşılaştırılması",
        labels={'Metric': 'Değerlendirme Metrikleri', 'Score': 'Skor Değeri', 'Optimization Goal': 'Optimizasyon Hedefi'}
    )
    fig.update_layout(
        yaxis=dict(range=[0, 0.9]),
        legend_title_text='Optimizasyon Hedefi',
        template='plotly_white'
    )

    return fig

def get_model_results():
    model_path = "Results/Test/BestParams/stacking/best_model_ROC-AUC.pkl"

    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
        
    # 2. Test verisini projeden çek
    _, X_test, _, y_test = extract_data(apply_target_encode=True, random_state=42)
    
    # 3. Model üzerinden tahminleri üret
    y_pred = loaded_model.predict(X_test)
    y_pred_proba = loaded_model.predict_proba(X_test)[:, 1]
    
    # 4. Metrikleri hesapla
    auc_val = roc_auc_score(y_test, y_pred_proba)
    f1_val = f1_score(y_test, y_pred)
    logloss_val = log_loss(y_test, y_pred_proba)
    precision_val = precision_score(y_test, y_pred, zero_division=0)
    recall_val = recall_score(y_test, y_pred, zero_division=0)

    return y_test, y_pred, y_pred_proba, auc_val, f1_val, logloss_val, precision_val, recall_val

def get_progress(val):
    return min(max(val, 0.0), 1.0) * 100

def get_html_code(auc_val, f1_val, logloss_val, precision_val, recall_val):
    html_code = f"""
    <style>
    .metric-container {{
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
        margin: 20px 0;
    }}
    .metric-circle {{
        width: 130px;
        height: 130px;
        border-radius: 50%;
        background: conic-gradient(#1f77b4 var(--p), #e9ecef var(--p) 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }}
    .metric-circle:hover {{
        transform: scale(1.05);
    }}
    .metric-circle::before {{
        content: '';
        position: absolute;
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: #ffffff;
    }}
    .metric-content {{
        position: relative;
        z-index: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }}
    .metric-title {{
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 2px;
        color: #495057;
    }}
    .metric-value {{
        font-size: 14px;
        font-weight: bold;
        color: #1f77b4;
    }}
    </style>

    <div class="metric-container">
        <div class="metric-circle" style="--p: {get_progress(auc_val)}%;">
            <div class="metric-content">
                <div class="metric-title">ROC-AUC</div>
                <div class="metric-value">{auc_val:.4f}</div>
            </div>
        </div>
        <div class="metric-circle" style="--p: {get_progress(f1_val)}%;">
            <div class="metric-content">
                <div class="metric-title">F1-Score</div>
                <div class="metric-value">{f1_val:.4f}</div>
            </div>
        </div>
        <div class="metric-circle" style="--p: {get_progress(logloss_val)}%;">
            <div class="metric-content">
                <div class="metric-title">Log Loss</div>
                <div class="metric-value">{logloss_val:.4f}</div>
            </div>
        </div>
        <div class="metric-circle" style="--p: {get_progress(precision_val)}%;">
            <div class="metric-content">
                <div class="metric-title">Precision</div>
                <div class="metric-value">{precision_val:.4f}</div>
            </div>
        </div>
        <div class="metric-circle" style="--p: {get_progress(recall_val)}%;">
            <div class="metric-content">
                <div class="metric-title">Recall</div>
                <div class="metric-value">{recall_val:.4f}</div>
            </div>
        </div>
    </div>
    """
    return html_code

def two_metric_circles(metric1_name, metric1_val, metric2_name, metric2_val):
    html_code = f"""
    <style>
    .metric-container {{
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
        margin: 20px 0;
    }}
    .metric-circle {{
        width: 130px;
        height: 130px;
        border-radius: 50%;
        background: conic-gradient(#1f77b4 var(--p), #e9ecef var(--p) 100%);
        display: flex;
        justify-content: center;
        align-items: center;
        position: relative;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }}
    .metric-circle:hover {{
        transform: scale(1.05);
    }}
    .metric-circle::before {{
        content: '';
        position: absolute;
        width: 110px;
        height: 110px;
        border-radius: 50%;
        background: #ffffff;
    }}
    .metric-content {{
        position: relative;
        z-index: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }}
    .metric-title {{
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 2px;
        color: #495057;
    }}
    .metric-value {{
        font-size: 14px;
        font-weight: bold;
        color: #1f77b4;
    }}
    </style>

    <div class="metric-container">
        <div class="metric-circle" style="--p: {get_progress(metric1_val)}%;">
            <div class="metric-content">
                <div class="metric-title">{metric1_name}</div>
                <div class="metric-value">{metric1_val:.4f}</div>
            </div>
        </div>
        <div class="metric-circle" style="--p: {get_progress(metric2_val)}%;">
            <div class="metric-content">
                <div class="metric-title">{metric2_name}</div>
                <div class="metric-value">{metric2_val:.4f}</div>
            </div>
        </div>
    </div>
    """
    return html_code

def barPlot_dataset_plotly(df, columnName, fig_width=10):
    column_counts = df[columnName].value_counts().reset_index()
    column_counts.columns = [columnName, 'Count']
    column_counts[columnName] = column_counts[columnName].astype(str)
    fig = px.bar(
        column_counts, 
        x=columnName, 
        y='Count',
        title=f'Count Plot of {columnName.capitalize()}',
        labels={columnName: columnName.capitalize(), 'Count': 'Count'},
        color='Count',
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        template='plotly_white',
        height=500
    )
    return fig

def univariate_analysis_plotly(df, column):
    ct = pd.crosstab(df[column], df["Y"], normalize="index") * 100
    ct = ct.reset_index()
    ct.columns = [str(col) for col in ct.columns]
    y_cols = [col for col in ct.columns if col != str(column)]
    ct_melted = ct.melt(id_vars=[str(column)], value_vars=y_cols, 
                        var_name='Accepted (Y)', value_name='Percentage')

    fig = px.bar(
        ct_melted,
        x=str(column),
        y='Percentage',
        color='Accepted (Y)',
        title=f"Percentage Acceptance by {column.capitalize()}",
        labels={str(column): column.capitalize(), 'Percentage': 'Percentage (%)'},
        barmode='stack',
        text_auto='.1f', # Barların üzerine doğrudan yüzde değerlerini yazar
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        template='plotly_white',
        height=550,
        legend_title="Accepted (Y)"
    )
    fig.update_traces(texttemplate='%{y:.1f}%', textposition='inside')
    
    return fig

def calculate_cramers_v(series1, series2):
    confusion_matrix = pd.crosstab(series1, series2)
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape

    if r <= 1 or k <= 1 or n == 0:
        return 0.0

    return np.sqrt((chi2 / n) / min(k - 1, r - 1))

def plot_cramers_v_matrix_plotly(df):
    columns = df.columns
    cramers_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for col1 in columns:
        for col2 in columns:
            cramers_matrix.loc[col1, col2] = calculate_cramers_v(df[col1], df[col2])

    cramers_matrix = cramers_matrix.astype(float)
    fig = px.imshow(
        cramers_matrix,
        text_auto=".2f",  # Hücrelerin üzerine skorları yazar
        color_continuous_scale="Blues",
        aspect="auto",
        title="Cramér's V Association Matrix of Categorical Variables"
    )
    fig.update_layout(
        xaxis_title="Features",
        yaxis_title="Features",
        xaxis_tickangle=-45,
        template="plotly_white",
        height=700
    )
    return fig