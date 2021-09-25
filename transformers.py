from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import pickle
import os
import abc
from helpers import *


def caching(transformer_function):
    def helper(*args, **kwargs):
        file_path = f'features/{args[0].__class__.__name__}_{str(args[1].shape[0])}.pkl'
        if os.path.isfile(file_path):
            return pd.read_pickle(file_path)

        with open(file_path, 'wb') as f:
            df = transformer_function(*args, **kwargs)
            pickle.dump(df, f)
            return df

    return helper


class BaseTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    @abc.abstractmethod
    def transform(self, X, y=None):
        pass


class CorrectByUserTransformer(BaseTransformer):

    def __init__(self, ):
        self.user_data = None

    def fit(self, X, y=None):
        self.user_data = filter_df_to_questions_only(X).groupby('user_id').agg(
            {'answered_correctly': ['count', 'mean']}).reset_index()
        self.user_data.columns = ['user_id', 'user_count', 'user_mean']
        return self

    @caching
    def transform(self, X, y=None):
        trans_X = X.merge(self.user_data, on='user_id', how='left')
        trans_X['user_count'].fillna(0, inplace=True)
        trans_X['user_mean'].fillna(0.5, inplace=True)
        return trans_X[['user_count', 'user_mean']]


class CorrectByQuestionTransformer(BaseTransformer):

    def __init__(self, ):
        self.question_data = None

    def fit(self, X, y=None):
        self.question_data = filter_df_to_questions_only(X).groupby('content_id').agg(
            {'answered_correctly': ['count', 'mean']}).reset_index()
        self.question_data.columns = ['content_id', 'question_count', 'question_mean']
        return self

    @caching
    def transform(self, X, y=None):
        trans_X = X.merge(self.question_data, on='content_id', how='left')
        trans_X['question_count'].fillna(0, inplace=True)
        trans_X['question_mean'].fillna(0.5, inplace=True)
        return trans_X[['question_count', 'question_mean']]


class CorrectByTagGroupTransformer(BaseTransformer):

    def __init__(self, questions):
        self.questions = questions
        self.tags_data = None

    def fit(self, X, y=None):
        X_with_tags = X.merge(self.questions, on='content_id', how='left')
        self.tags_data = X_with_tags.groupby('tag_group').agg({'answered_correctly': ['count', 'mean']}).reset_index()
        self.tags_data.columns = ['tag_group', 'question_tag_count', 'question_tag_mean']
        return self

    @caching
    def transform(self, X, y=None):
        X_with_tags = X.merge(self.questions, on='content_id', how='left')
        trans_X = X_with_tags.merge(self.tags_data, on='tag_group', how='left')
        trans_X['question_tag_count'].fillna(0, inplace=True)
        trans_X['question_tag_mean'].fillna(0.5, inplace=True)
        return trans_X[['question_tag_count', 'question_tag_mean']]
