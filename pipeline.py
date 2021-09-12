import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid
from questions_tags_clustering import get_questions_tag_groups
from transformers import *
from lightgbm import LGBMClassifier
import optuna
from hyperparameters import *


def predict_classification(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    # scores = {"accuracy": accuracy_score(y_test, y_pred),
    #           "f1 score": f1_score(y_test, y_pred),
    #           "precision": precision_score(y_test, y_pred),
    #           "recall": recall_score(y_test, y_pred),
    #           "confusion matrix": confusion_matrix(y_test, y_pred)}
    scores = roc_auc_score(y_test, y_pred)
    return scores


def filter_data(df, percentage=0.00001):
    users = df['user_id'].unique()
    print(int(percentage * df.shape[0]))
    filtered_df = df.set_index('user_id').loc[users[:int(percentage * df.shape[0])]].reset_index()
    return filtered_df[filtered_df['answered_correctly'] != -1]


def cross_validate(pipeline, path='archive/', target='answered_correctly', n_folds=5):
    scores_val, scores_train = [], []
    for ind in range(1, n_folds + 1):
        train = filter_data(pd.read_pickle(f'{path}cv{ind}_train.pickle'))
        val = filter_data(pd.read_pickle(f'{path}cv{ind}_valid.pickle'))

        pipeline.fit(train, train[target], eval_metric='logloss')
        scores_val.append(predict_classification(pipeline, val.drop(target, axis=1), val[target]))
        scores_train.append(predict_classification(pipeline, train.drop(target, axis=1), train[target]))

    return dict(pd.DataFrame(scores_train).mean()), dict(pd.DataFrame(scores_val).mean())


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

    scores_train, scores_test = cross_validate(pipeline)
    return scores_test[0]


def model_search():
    ensemble_clf = [XGBClassifier, LGBMClassifier]

    for i in range(len(ensemble_clf)):
        objective = lambda trial: optuna_search(trial, ensemble_clf[i], get_xgboost_params)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=1)  # 10
        print(ensemble_clf[i])
        print(study.best_trial)
        print(study.best_value)


def grid_search():
    transformer_list = create_transformers()

    param_options = {'n_estimators': [100],
                     'objective': ['binary'],
                     'is_unbalance': [True],
                     }
    best_score = 0
    best_option = None

    for option in ParameterGrid(param_options):
        pipeline = Pipeline([('transformers', FeatureUnion(transformer_list)),
                             ('model', LGBMClassifier(**option))])

        scores_train, scores_test = cross_validate(pipeline)
        print("train scores: ", scores_train)
        print("test scores: ", scores_test)
        best_score = max(scores_test, best_score)
        best_option = option if best_score == scores_test else best_option
    return best_score, best_option


if __name__ == '__main__':
    model_search()
