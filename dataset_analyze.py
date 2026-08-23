import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
from scipy.stats import chi2_contingency
wr.filterwarnings('ignore')

df = pd.read_csv("Dataset/preprocessed_dataset.csv")

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

def bivariate_analysis(column):
  ct = pd.crosstab(df[column], df['Y'], normalize='index') * 100

  ct.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Set2")
  plt.title(f"Percentage Acceptance by {column.capitalize()}")
  plt.xlabel(column.capitalize())
  plt.ylabel("Percentage (%)")
  plt.legend(title="Accepted (Y)", bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.xticks(rotation=45, ha="right")
  plt.tight_layout()
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
    
    for column in df.columns:
        #bivariate_analysis(column)
        #barPlot_dataset(column)
        calculate_missing_lift('Restaurant20To50',column)
        #Bar,CoffeeHouse,CarryAway,RestaurantLessThan20,Restaurant20To50



if __name__ == "__main__":
    main()