import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gui_helper as gh
from gui_helper import *
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_auc_score, f1_score, log_loss, precision_score, recall_score,
    confusion_matrix, roc_curve
)
import pickle
from comparing import extract_data

st.set_page_config(page_title="In-Vehicle Coupon Recommendation", layout="wide")
st.title("In-Vehicle Coupon Recommendation - Makine Öğrenmesi")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Keşifsel Veri Analizi (EDA)", "🏆 Model Karşılaştırmaları", "⚙️ Optuna Optimizasyon Sonuçları", "🎯 Sonuç", "🧬 Sentetik Veri Sonuçları"])

with tab1:
    st.header("Keşifsel Veri Analizi ve Önemli Bulgular")
    st.write("Veri setinin ilk 5 satırı:")
    st.dataframe(gh.df.head(), use_container_width=True)
    st.markdown("---")

    st.write("Veri setinin istatiksel özeti:")
    st.dataframe(gh.df.describe().T, use_container_width=True)
    st.write("Bu özette toCoupon_GEQ5min sütunun varyansının 0 olduğu " \
        "görülmektedir ve bu, sütunun bir bilgi içermediği anlamına gelir. " \
        "Ayrıca yön sütunları aynı varyansa sahiptir ve ortalamaları 1'e tamamlanmaktadır." \
        "Bu ise bu iki sütunun aynı bilgiyi içerdiği anlamına gelir.")
    st.markdown("---")

    missing_count = gh.df.isnull().sum()
    missing_percent = (gh.df.isnull().mean() * 100).round(2)
    df_missing = pd.DataFrame({
        'Eksik Değer Sayısı': missing_count,
        'Eksik Oranı (%)': missing_percent
    })
    df_missing_filtered = df_missing[df_missing['Eksik Değer Sayısı'] > 0]

    st.write("Veri setinde eksik değer içeren öznitelikler:")
    st.dataframe(df_missing_filtered, use_container_width=True)
    
    # Görsel Isı Haritası (Heatmap)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(gh.df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
    ax.set_title("Eksik Değerlerin Dağılımı (Sarı Alanlar Eksik Değerleri Gösterir)", fontsize=12)
    st.pyplot(fig)
    st.markdown("---")

    st.write("Kuponun kabul edildiğinin/edilmediğinin dağılımı:")
    st.markdown(two_metric_circles("0 - Kabul Edilmedi", 0.43156733, "1 - Kabul Edildi", 0.56843267), unsafe_allow_html=True)
    st.markdown("---")

    st.write("Veri setindeki kategorik değişkenlerin dağılım grafikleri:")
    selected_column = st.selectbox(
        "Grafiği çizilecek sütunu seçin:", 
        gh.df.select_dtypes(include=['object', 'int64']).columns,
        key='bar_plot_select'
    )
    if selected_column:
        fig_plotly = barPlot_dataset_plotly(gh.df, selected_column)
        st.plotly_chart(fig_plotly, use_container_width=True)
    st.markdown("---")

    st.write("Kategorik değişkenlerin hedef değişken (Y) üzerindeki yüzdesel yığılmış dağılımı:")
    selected_col = st.selectbox(
        "Analiz edilecek sütunu seçin:", 
        gh.df.select_dtypes(include=['object', 'int64']).columns,
        key='univariate_select'
    )
    if selected_col:
        fig_uni = univariate_analysis_plotly(gh.df, selected_col)
        st.plotly_chart(fig_uni, use_container_width=True)
    st.markdown("---")

    st.write("Veri setindeki kategorik özniteliklerin birbirleriyle olan istatistiksel ilişki düzeyleri:")
    fig_cramers = plot_cramers_v_matrix_plotly(gh.df)
    st.plotly_chart(fig_cramers, use_container_width=True)

with tab2:
    st.header("Model Başarım Karşılaştırmaları")
    st.write("Farklı algoritmaların farklı veri seti varyasyonları üzerindeki sonuçları:")
    fig = heatmap_comparison()
    st.pyplot(fig)
    

with tab3:
    st.header("Hiperparametre Optimizasyonu (Optuna)")
    st.write("LightGBM, Bagging, SVM ve Stacking modelleri için 5-Fold Stratified CV ile yapılan Optuna optimizasyon sonuçları:")
    
    st.subheader("Stacking")
    fig = plot_manual_results("stacking")
    st.plotly_chart(fig)
    st.subheader("LightGBM")
    fig = plot_manual_results("lgbm")
    st.plotly_chart(fig)
    st.subheader("Bagging")
    fig = plot_manual_results("bagging")
    st.plotly_chart(fig)
    st.subheader("Support Vector Machines")
    fig = plot_manual_results("svm")
    st.plotly_chart(fig)

with tab4:
    st.header("Son Modelin Sonuçları ve Performans Detayları")

    y_test,y_pred,y_pred_proba,auc_val,f1_val,logloss_val,precision_val,recall_val = get_model_results("Results/Test/BestParams/stacking/best_model_ROC-AUC.pkl")
    
    # --- 5 TANE YAN YANA DAİRE KISMI ---
    html_code = get_html_code(auc_val,f1_val,logloss_val,precision_val,recall_val)
    st.markdown(html_code, unsafe_allow_html=True)
    # ----------------------------------
    
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Confusion Matrix (Hata Matrisi)")
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = px.imshow(
            cm, 
            text_auto=True, 
            color_continuous_scale="Blues",
            labels=dict(x="Tahmin Edilen Sınıf", y="Gerçek Sınıf", color="Adet"),
            x=['Kabul Etmez (0)', 'Kabul Eder (1)'],
            y=['Kabul Etmez (0)', 'Kabul Eder (1)']
        )
        fig_cm.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_cm, width='stretch')
        
    with chart_col2:
        st.subheader("ROC Eğrisi (ROC Curve)")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, 
            mode='lines+markers', 
            name=f'Model ROC (AUC = {auc_val:.4f})', 
            line=dict(width=3, color='#1f77b4')
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], 
            mode='lines', 
            name='Rastgele Tahmin', 
            line=dict(dash='dash', color='gray')
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity)",
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_roc, width='stretch')

with tab5:
    st.header("Sentetik Veri İle Eğitilen Modelin Sonuçları ve Performans Detayları")

    y_test,y_pred,y_pred_proba,auc_val,f1_val,logloss_val,precision_val,recall_val = get_model_results("Results/Test/BestParams/stacking/best_model_ROC-AUC-SYNTHETIC.pkl")
    
    # --- 5 TANE YAN YANA DAİRE KISMI ---
    html_code = get_html_code(auc_val,f1_val,logloss_val,precision_val,recall_val)
    st.markdown(html_code, unsafe_allow_html=True)
    # ----------------------------------
    
    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Confusion Matrix (Hata Matrisi)")
        cm = confusion_matrix(y_test, y_pred)
        
        fig_cm = px.imshow(
            cm, 
            text_auto=True, 
            color_continuous_scale="Blues",
            labels=dict(x="Tahmin Edilen Sınıf", y="Gerçek Sınıf", color="Adet"),
            x=['Kabul Etmez (0)', 'Kabul Eder (1)'],
            y=['Kabul Etmez (0)', 'Kabul Eder (1)']
        )
        fig_cm.update_layout(
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_cm, width='stretch')
        
    with chart_col2:
        st.subheader("ROC Eğrisi (ROC Curve)")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, 
            mode='lines+markers', 
            name=f'Model ROC (AUC = {auc_val:.4f})', 
            line=dict(width=3, color='#1f77b4')
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], 
            mode='lines', 
            name='Rastgele Tahmin', 
            line=dict(dash='dash', color='gray')
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity)",
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_roc, width='stretch')