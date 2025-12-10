import pandas as pd

def load_data(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def align_dates(df1, df2):
    common = df1.index.intersection(df2.index)
    return df1.loc[common], df2.loc[common]
