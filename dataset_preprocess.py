import pandas as pd
import numpy as np

df = pd.read_csv("Dataset/in-vehicle-coupon-recommendation.csv")

#DROP COLUMNS
df = df.drop(columns=['car', 'direction_same', 'toCoupon_GEQ5min'])

#FILL MISSING VALUES
missing_columns = df.columns[df.isnull().any()]
for col in missing_columns:
    mod_deger = df[col].mode()[0]
    df[col] = df[col].fillna(mod_deger)
    print(f"'{col}' sütunundaki eksikler '{mod_deger}' ile dolduruldu.")

df.to_csv("Dataset/preprocessed_dataset.csv", index=False)