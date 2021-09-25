from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from questions_tags_clustering import get_questions_tag_groups
from transformers import *
from lightgbm import LGBMClassifier
import optuna
from hyperparameters import *
import numpy as np


def predict_classification(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return roc_auc_score(y_test, y_pred)


def cross_validate(pipeline, path='archive/', target='answered_correctly', n_folds=5):
    scores_val = []
    for ind in range(1, n_folds + 1):
        train = filter_df_to_questions_only(pd.read_pickle(f'{path}cv{ind}_train.pickle'))
        val = filter_df_to_questions_only(pd.read_pickle(f'{path}cv{ind}_valid.pickle'))

        pipeline.fit(train, train[target])
        scores_val.append(predict_classification(pipeline, val.drop(target, axis=1), val[target]))

    return np.mean(scores_val)


def read_questions_df(path='data/'):
    questions = pd.read_csv(f'{path}questions.csv')
    questions.rename({'question_id': 'content_id'}, axis=1, inplace=True)
    return questions


def create_transformers():
    questions = read_questions_df()
    questions['tag_group'] = get_questions_tag_groups(questions)

    transformer_list = [('user_correct_answers_count', CorrectByUserTransformer()),
                        ('question_correct_answers_count', CorrectByQuestionTransformer()),
                        ('question_tags', CorrectByTagGroupTransformer(questions))]
    return transformer_list


def optuna_search(trial, estimator, param_generator):
    model = estimator(**param_generator(trial))

    transformer_list = create_transformers()
    pipeline = Pipeline([('transformers', FeatureUnion(transformer_list)),
                         ('model', model)])

    scores_test = cross_validate(pipeline)
    return scores_test


def model_search():
    models = {'xgboost': (XGBClassifier, get_xgboost_params),
              'lgbm': (LGBMClassifier, get_lgbm_params),
              'random_forest': (RandomForestClassifier, get_random_forest_params),
              'logistic_regression': (LogisticRegression, get_logistic_regression_params)}

    for model_name, model_conf in models.items():
        clf, params = model_conf
        objective = lambda trial: optuna_search(trial, clf, params)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        print(f'Best trial for classifier {model_name} is {study.best_trial} with roc-auc score of {study.best_value} ')


if __name__ == '__main__':
    model_search()
