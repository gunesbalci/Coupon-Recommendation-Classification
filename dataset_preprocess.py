import pandas as pd
import numpy as np

df = pd.read_csv("Dataset/in-vehicle-coupon-recommendation.csv")

#DROP DUPLICATE ROWS
df = df.drop_duplicates()
#DROP COLUMNS
df = df.drop(columns=['car', 'direction_same', 'toCoupon_GEQ5min'])

#FILL MISSING VALUES
missing_columns = df.columns[df.isnull().any()]
for col in missing_columns:
    mod_deger = df[col].mode()[0]
    df[col] = df[col].fillna(mod_deger)
    print(f"Missing values in column '{col}' filled with '{mod_deger}'.")

#NEW FEATURES - to_coupon 
conditions = [
    (df["toCoupon_GEQ15min"] == 0) & (df["toCoupon_GEQ25min"] == 0),
    (df["toCoupon_GEQ15min"] == 1) & (df["toCoupon_GEQ25min"] == 0),
    (df["toCoupon_GEQ25min"] == 1),
]
choices = [0, 1, 2]
df["to_coupon"] = np.select(conditions, choices, default=-1)
df = df.drop(columns=['toCoupon_GEQ15min', 'toCoupon_GEQ25min'])

#NEW FEATURES - coupon_frequency
conditions = [
    (df["coupon"] == "Restaurant(<20)"),
    (df["coupon"] == "Coffee House"),
    (df["coupon"] == "Carry out & Take away"),
    (df["coupon"] == "Bar"),
    (df["coupon"] == "Restaurant(20-50)")
]
choices = [df["RestaurantLessThan20"], df["CoffeeHouse"], df["CarryAway"], df["Bar"], df["Restaurant20To50"]]
df["coupon_frequency"] = np.select(conditions, choices, default=np.nan)
df = df.drop(columns=['RestaurantLessThan20', 'CoffeeHouse', 'CarryAway', 'Bar', 'Restaurant20To50'])

df.to_csv("Dataset/preprocessed_dataset.csv", index=False)