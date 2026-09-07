from train_models import *
import warnings as wr

wr.filterwarnings('ignore')

# ** This functions trains stacking model.
# best_params: Optimization of metric of choice is possible. Example: best_params=modelname_best_metric_params
# model_type: 4 different models can be selected. 'lgbm', 'bagging', 'svm', 'stacking'.
# best_metric: Selected metric in text.
# save: Whether save trained model file or not.
train_model(stacking_best_roc_auc_params, model_type='stacking', best_metric="ROC-AUC", save=True)