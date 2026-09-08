from train_models import *
import warnings as wr
from dataset_preprocess import create_train_test_datasets
from dataset_encode import encode_train_test_datasets

wr.filterwarnings('ignore')

# Upload dataset
df = pd.read_csv("Dataset/in-vehicle-coupon-recommendation.csv")
print("Dataset is pulled.")

# ** This function applies preprocess. Removes duplicates. Splits data into train and test sets (80:20). 
#    Drops columns with no info. Fills missing values with mod. Creates 2 new features.
# data: dataset to apply preprocess.
# print_on: whether to print result of filling missing values. (defult: false)
# random_state: split seed. (default: 42)
X_train, X_test, y_train, y_test = create_train_test_datasets(df)
print("Preprocess operations are done.")

# ** This function encodes input data. Applies ordinal encoding to ordinal data,
#    one-hot encoding to nominal data and target encoding to column "occupation".
#    Also "expiration" and "gender" columns encoded to be binary.
# X_train: Input train data.
# X_test: Input test data.
# Y_train: Output train data.
# apply_target_encode: Target encoding is applied to "occupation" if true. If not one-hot encoding is applied.
X_train, X_test = encode_train_test_datasets(X_train, X_test, y_train, True)
print("Encoding operations are done.")

data = X_train, X_test, y_train, y_test

# ** This function trains the model selected with given data.
# data: data to train the model with.
# best_params: Optimization of metric of choice is possible. Example: best_params=modelname_best_metric_params
# model_type: 4 different models can be selected. 'lgbm', 'bagging', 'svm', 'stacking'.
# best_metric: Selected metric in text.
# save: Whether save trained model file or not.
train_model(data, stacking_best_roc_auc_params, model_type='stacking', best_metric="ROC-AUC", save=True)
print("Model training is complete!")