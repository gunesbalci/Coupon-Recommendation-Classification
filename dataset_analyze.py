import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
from scipy.stats import chi2_contingency
wr.filterwarnings('ignore')

df = pd.read_csv("Dataset/in-vehicle-coupon-recommendation.csv")

def analyze_dataset():
    print(df.head()) #top 5 rows of the dataset
    print("Shape of the dataset: " + str(df.shape))
    df.info() 
    print(df.describe().T) #statistical summary of the dataset
    print("Missing values:\n" + str(df.isnull().sum())) #check for missing values
    print("Duplicate values: " + str(df.duplicated().sum())) #check for duplicate values
    print("Target Distribution (Y):\n" + str(df['Y'].value_counts(normalize=True) * 100))

def barPlot_dataset(columnName,fig_width=10):
    column_counts = df[columnName].value_counts()

    if df[columnName].dtype == 'int64':
        raw_labels = [str(val) for val in column_counts.sort_index().index]
    else:
        raw_labels = column_counts.index

    # character limit for x-axis labels
    num_categories = len(raw_labels)
    if num_categories > 0:
        max_chars = max(
            3, int((fig_width * 10) / num_categories)) 
    else:
        max_chars = 10
        
    x_labels = []
    for label in raw_labels:
        label_str = str(label)
        if len(label_str) > max_chars:
            x_labels.append(label_str[:max_chars] + "...")
        else:
            x_labels.append(label_str)

    plt.figure(figsize=(10, 8))
    plt.bar(x_labels, column_counts, color="blue")
    plt.title(f'Count Plot of {columnName.capitalize()}')
    plt.xlabel(f'{columnName.capitalize()}')
    plt.ylabel('Count')
    plt.show()

def univariate_analysis(column, save_path="Results/Version1/Bivariate"):
    # Çapraz tabloyu yüzdelik olarak hesapla
    ct = pd.crosstab(df[column], df["Y"], normalize="index") * 100

    # 1. Figür ve eksenleri açıkça tanımlıyoruz (fazladan figür oluşmasını engeller)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Grafiği ax üzerine çizdiriyoruz
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")

    plt.title(f"Percentage Acceptance by {column.capitalize()}")
    plt.xlabel(column.capitalize())
    plt.ylabel("Percentage (%)")
    plt.legend(title="Accepted (Y)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")

    # --- Barların Üzerine Yüzdelik Değerleri Yazdırma (0 ve 1 için) ---
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f%%",
            label_type="center",
            fontsize=9,
            color="black",
            weight="bold",
            padding=3,
        )

    # Otomatik klasör oluşturma ve kaydetme
    os.makedirs(save_path, exist_ok=True)
    file_name = f"{save_path}/acceptance_by_{column}.png"

    plt.tight_layout()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    print(f"Grafik kaydedildi: {file_name}")

    # Göster ve hafızayı temizle (fazladan boş figür çıkmasını önler)
    plt.show()
    plt.close(fig)

import os
import matplotlib.pyplot as plt
import seaborn as sns


def faceted_bivariate_analysis(
    x_col, hue_col, save_path="Results/Analyze/Version1/Faceted_Bivariate"):

  g = sns.catplot(
      data=df,
      x=x_col,
      hue=hue_col,
      col="Y",  # Y sütununa göre grafikleri yan yana ayırır (Y=0 ve Y=1)
      kind="count",  # Sayım grafiği (count plot)
      height=5,
      aspect=1,
      palette="Set2",
  )

  # Grafik başlıklarını ve etiketlerini düzenleme
  g.set_axis_labels(x_col.capitalize(), "Count")
  g.set_titles(col_template="Y = {col_name}")
  g.fig.subplots_adjust(top=0.85)  # Üst başlık için pay bırak
  g.fig.suptitle(
      f"Count of {x_col.capitalize()} by {hue_col.capitalize()} split by Acceptance (Y)",
      fontsize=14,
  )

  for ax in g.axes.flat:
    for label in ax.get_xticklabels():
      label.set_rotation(45)
      label.set_ha("right")

  os.makedirs(save_path, exist_ok=True)
  file_name = f"{save_path}/count_{x_col}_by_{hue_col}_split_by_Y.png"

  g.savefig(file_name, dpi=300, bbox_inches="tight")
  print(f"Çoklu grafik kaydedildi: {file_name}")

  plt.show()

def calculate_cramers_v(series1, series2):
  confusion_matrix = pd.crosstab(series1, series2)

  chi2, _, _, _ = chi2_contingency(confusion_matrix)
  n = confusion_matrix.sum().sum()
  r, k = confusion_matrix.shape

  if r <= 1 or k <= 1 or n == 0:
    return 0.0

  return (chi2 / n) / min(k - 1, r - 1)

def plot_cramers_v_matrix():
  columns = df.columns

  cramers_matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
  for col1 in columns:
    for col2 in columns:
      cramers_matrix.loc[col1, col2] = calculate_cramers_v(df[col1], df[col2])

  cramers_matrix = cramers_matrix.astype(float)

  plt.figure(figsize=(12, 10))
  sns.heatmap(
      cramers_matrix,
      annot=False,
      cmap="Blues",
      cbar=True,
      linewidths=0.5,
  )  
  plt.title("Cramér's V Association Matrix of Categorical Variables")
  plt.xticks(rotation=45, ha="right")
  plt.yticks(rotation=0)
  plt.tight_layout()
  plt.show()

def calculate_missing_lift(target_column, feature_to_check):
  # 1. Boş bırakanların yüzdelik dağılımı
  missing_subset = df[df[target_column].isnull()]
  missing_dist = missing_subset[feature_to_check].value_counts(normalize=True) * 100

  # 2. Tüm veri setinin yüzdelik dağılımı
  overall_dist = df[feature_to_check].value_counts(normalize=True) * 100

  # 3. İkisini birleştirip "Boş / Genel" oranını (Lift) hesaplayalım
  comparison = pd.DataFrame(
      {
          "Boş Bırakanlar (%)": missing_dist,
          "Tüm Veri Seti (%)": overall_dist,
      }
  )

  # Orantılama (Boş bırakan yüzde / Genel yüzde)
  comparison["Oran (Lift)"] = (
      comparison["Boş Bırakanlar (%)"] / comparison["Tüm Veri Seti (%)"]
  )

  print(
      f"--- '{target_column}' Eksikliği ve '{feature_to_check}' Orantılama"
      " Analizi ---"
  )
  print(round(comparison, 2))
  print("\n" + "=" * 50 + "\n")

def main():
    #analyze_dataset()
    #plot_cramers_v_matrix()
    
    #for column in df.columns:
        #univariate_analysis(column)
        #barPlot_dataset(column)
        #calculate_missing_lift('Restaurant20To50',column)
        #Bar,CoffeeHouse,CarryAway,RestaurantLessThan20,Restaurant20To50

    #faceted_bivariate_analysis(x_col="destination", hue_col="time")
    #faceted_bivariate_analysis(x_col="destination", hue_col="passanger")
    #faceted_bivariate_analysis(x_col="time", hue_col="coupon")
    #faceted_bivariate_analysis(x_col="weather", hue_col="temperature")
    #faceted_bivariate_analysis(x_col="expiration", hue_col="coupon")
    #faceted_bivariate_analysis(x_col="income", hue_col="coupon")
    faceted_bivariate_analysis(x_col="time", hue_col="has_children")

if __name__ == "__main__":
    main()