import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("Dataset/in-vehicle-coupon-recommendation.csv")

def drop_columns(data):
    #DROP COLUMNS
    data = data.drop(columns=['car', 'direction_same', "direction_opp", 'toCoupon_GEQ5min'])
    return data

def fill_missing_values(data,data_opt,print_on=False):
    missing_columns = data.columns[data.isnull().any()]
    for col in missing_columns:
        mod_deger = data[col].mode()[0]
        data[col] = data[col].fillna(mod_deger)
        if data_opt is not None:
            data_opt[col] = data_opt[col].fillna(mod_deger)
        if print_on:
            print(f"Missing values in column '{col}' filled with '{mod_deger}'.")
    return data, data_opt

def create_new_features(data):
    #NEW FEATURES - to_coupon 
    conditions = [
        (data["toCoupon_GEQ15min"] == 0) & (data["toCoupon_GEQ25min"] == 0),
        (data["toCoupon_GEQ15min"] == 1) & (data["toCoupon_GEQ25min"] == 0),
        (data["toCoupon_GEQ25min"] == 1),
    ]
    choices = [0, 1, 2]
    data["to_coupon"] = np.select(conditions, choices, default=-1)
    data = data.drop(columns=['toCoupon_GEQ15min', 'toCoupon_GEQ25min'])

    #NEW FEATURES - coupon_frequency
    conditions = [
        (data["coupon"] == "Restaurant(<20)"),
        (data["coupon"] == "Coffee House"),
        (data["coupon"] == "Carry out & Take away"),
        (data["coupon"] == "Bar"),
        (data["coupon"] == "Restaurant(20-50)")
    ]
    choices = [data["RestaurantLessThan20"], data["CoffeeHouse"], data["CarryAway"], data["Bar"], data["Restaurant20To50"]]
    data["coupon_frequency"] = np.select(conditions, choices, default=np.nan)
    data = data.drop(columns=['RestaurantLessThan20', 'CoffeeHouse', 'CarryAway', 'Bar', 'Restaurant20To50'])
    return data

def create_train_test_datasets(data, print_on=False):
    data = data.drop_duplicates()
    
    # TRAIN-TEST SPLIT
    X = data.drop(columns=["Y"])
    y = data["Y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train = drop_columns(X_train)
    X_test = drop_columns(X_test)
    X_train, X_test = fill_missing_values(X_train, X_test, print_on)
    X_train = create_new_features(X_train)
    X_test = create_new_features(X_test)

    return X_train, X_test, y_train, y_test

def combine_occupation_values(data):
    occupation_mapping = {
        'Architecture & Engineering': 'Other',
        'Building & Grounds Cleaning & Maintenance': 'Other',
        'Construction & Extraction': 'Other',
        'Farming Fishing & Forestry': 'Other',
        'Installation Maintenance & Repair': 'Other',
        'Life Physical Social Science': 'Other',
        'Personal Care & Service': 'Other',
        'Production Occupations': 'Other',
        'Protective Service': 'Other'
    }
    data['occupation'] = data['occupation'].replace(occupation_mapping)
    return data