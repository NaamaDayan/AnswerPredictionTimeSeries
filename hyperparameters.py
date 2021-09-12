def get_lgbm_params(trial):
    return {
        # "device_type": trial.suggest_categorical("device_type", ['gpu']),
        "n_estimators": trial.suggest_categorical("n_estimators", [150, 500, 1000]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "max_bin": trial.suggest_int("max_bin", 200, 300),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float(
            "bagging_fraction", 0.2, 0.95, step=0.1
        ),
        "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [2]),
        'objective': trial.suggest_categorical("objective", ['binary:logistic'])
    }


def get_xgboost_params(trial):
    return {'max_depth': trial.suggest_int("max_depth", 3, 8),
            'subsample': trial.suggest_loguniform("subsample", 0.3, 0.9),
            'n_estimators': trial.suggest_categorical("n_estimators", [150, 500, 1000]),
            'learning_rate': trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            'objective': trial.suggest_categorical("objective", ['binary:logistic']),
            'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [2]),
            'use_label_encoder': trial.suggest_categorical('use_label_encoder', [False])
            }


def get_random_forest_params(trial):
    return {'n_estimators': trial.suggest_categorical('n_estimators', [100, 300, 500, 800, 1200]),
            'max_depth': trial.suggest_categorical('max_depth', [5, 8, 15, 25, 30]),
            'min_samples_split': trial.suggest_categorical('min_samples_split', [2, 5, 10, 15, 100]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 5, 10]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced'])}


def get_logistic_regression_params(trial):
    return {'c_values': trial.suggest_categorical('c_values', [100, 10, 1.0, 0.1, 0.01]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced'])}
