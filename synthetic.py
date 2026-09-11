import dataxid
import pandas as pd
from dataset_preprocess import create_train_test_datasets, create_new_features
from dataset_encode import encode_train_test_datasets
from sklearn.metrics import normalized_mutual_info_score

def create_synthetic_data(data_name, generate=True, maxepochs=100, dropout=0.5,
    earlystoppatience=4, embeddingdim=64, sample_multiplier=1):

    dataxid.api_key = "dx_test_611aa1dd4ebd8aa5685aedc0a3459214687858805e6e3ea8"
    df = pd.read_csv("Dataset/in-vehicle-coupon-recommendation.csv")

    X_train, X_test, y_train, y_test = create_train_test_datasets(df,create_features=False)
    train_df = pd.concat([X_train, y_train], axis=1)

    if generate:
        model = dataxid.Model.create(
            data=train_df,
            config=dataxid.ModelConfig(
                model_size="medium",
                max_epochs=maxepochs,
                embedding_dropout=dropout,
                batch_size=128,
                early_stop_patience=earlystoppatience,
                embedding_dim=embeddingdim,
            ),
        )
        synthetic_1x = model.generate(n_samples=(10088*sample_multiplier))
        model.delete()
        synthetic_1x.to_csv(f"Dataset/synthetic_1x_{data_name}.csv", index=False)

    return X_test, y_test

def preprocess(X_train, X_test, y_train):
    X_train = create_new_features(X_train)
    X_test = create_new_features(X_test)
    X_train, X_test = encode_train_test_datasets(X_train, X_test, y_train, True)
    return X_train, X_test

def target_relationship_diff(real_df, synth_df, target_col):
    results = []
    for col in real_df.columns:
        if col == target_col:
            continue
        real_mi = normalized_mutual_info_score(real_df[col], real_df[target_col])
        synth_mi = normalized_mutual_info_score(synth_df[col], synth_df[target_col])
        
        results.append({
            "column": col,
            "real_mi_with_target": real_mi,
            "synth_mi_with_target": synth_mi,
            "mi_diff": abs(real_mi - synth_mi)
        })
    
    return pd.DataFrame(results).sort_values("mi_diff", ascending=False)