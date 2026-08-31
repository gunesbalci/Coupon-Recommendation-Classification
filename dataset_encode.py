import pandas as pd
import re
import category_encoders as ce

def ordinal_encode(data):
    temperature_map = {30: 0, 55: 1, 80: 2}
    data["temperature"] = data["temperature"].map(temperature_map)

    time_map = {"7AM": 0, "10AM": 1, "2PM": 2, "6PM": 3, "10PM": 4}
    data["time"] = data["time"].map(time_map)

    age_map = {"below21": 0, "21": 1, "26": 2, "31": 3, "36": 4, "41": 5, "46": 6, "50plus": 7}
    data["age"] = data["age"].map(age_map)

    education_map = {"Some High School": 0, "High School Graduate": 1,
        "Some college - no degree": 2, "Associates degree": 3, "Bachelors degree": 4,
        "Graduate degree (Masters or Doctorate)": 5}
    data["education"] = data["education"].map(education_map)

    income_map = {"Less than $12500": 0, "$12500 - $24999": 1, "$25000 - $37499": 2,
        "$37500 - $49999": 3, "$50000 - $62499": 4, "$62500 - $74999": 5,
        "$75000 - $87499": 6, "$87500 - $99999": 7, "$100000 or More": 8}
    data["income"] = data["income"].map(income_map)

    frequency_map = {"never": 0, "less1": 1, "1~3": 2, "4~8": 3, "gt8": 4}
    data["coupon_frequency"] = data["coupon_frequency"].map(frequency_map)
    return data

def one_hot_encode(x_train, x_test, apply_target_encode=True):
    if not apply_target_encode:
        encode_columns = ['destination', 'passanger', 'occupation',
                            'weather', 'coupon', 'maritalStatus']
    else:
        encode_columns = ['destination', 'passanger', 
                            'weather', 'coupon', 'maritalStatus']
    x_train = pd.get_dummies(x_train, dtype=int, columns=encode_columns)
    x_test = pd.get_dummies(x_test, dtype=int, columns=encode_columns)

    # Train ve test setlerindeki sütun isimlerini eşitle (Biri diğerinde eksik kalmasın)
    x_train, x_test = x_train.align(x_test, join="left", axis=1, fill_value=0)
    return x_train, x_test

def target_encode(x_train, x_test, y_train):
    target_encoder = ce.TargetEncoder(cols=["occupation"])

    # Sadece train ile fit_transform, test ile sadece transform!
    x_train["occupation"] = target_encoder.fit_transform(
        x_train["occupation"], y_train
    )
    x_test["occupation"] = target_encoder.transform(x_test["occupation"])
    return x_train, x_test

def fix_expiration_gender(data):
    # expiration sütunundaki değerleri 0 ve 1 olarak değiştir
    expiration_map = {"1d": 0, "2h": 1}
    data["expires_soon"] = data["expiration"].map(expiration_map)
    data = data.drop(columns=['expiration'])

    gender_map = {"Male": 0, "Female": 1}
    data["gender"] = data["gender"].map(gender_map)
    return data

def encode_train_test_datasets(x_train, x_test, y_train, apply_target_encode):
    x_train = ordinal_encode(x_train)
    x_test = ordinal_encode(x_test)
    x_train = fix_expiration_gender(x_train)
    x_test = fix_expiration_gender(x_test)

    x_train, x_test = one_hot_encode(x_train, x_test, apply_target_encode)
    if apply_target_encode:
        x_train, x_test = target_encode(x_train, x_test, y_train)

    x_train.columns = [re.sub(r"[\[\]<>]", "_", str(col)) for col in x_train.columns]
    x_test.columns = [re.sub(r"[\[\]<>]", "_", str(col)) for col in x_test.columns]
    return x_train, x_test