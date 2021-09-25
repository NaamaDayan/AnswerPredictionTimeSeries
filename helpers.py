
def filter_df_to_questions_only(df):
    return df[df['answered_correctly'] != -1]