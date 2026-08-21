import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

df = pd.read_csv("Dataset/in-vehicle-coupon-recommendation.csv")

def analyze_dataset():
    print(df.head()) #top 5 rows of the dataset
    print("Shape of the dataset: " + str(df.shape))
    df.info() 
    print(df.describe().T) #statistical summary of the dataset
    print("Missing values:\n" + str(df.isnull().sum())) #check for missing values
    print("Duplicate values: " + str(df.duplicated().sum())) #check for duplicate values

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

def main():
    analyze_dataset()

    for column in df.columns:
        barPlot_dataset(column)


if __name__ == "__main__":
    main()