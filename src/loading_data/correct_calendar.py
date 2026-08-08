import pandas as pd

def calendar_adjustment(dataframe: pd.DataFrame):
    dataframe = dataframe.copy()

    dataframe["calendar_month"] = ((dataframe["month"] - 3) % 12) + 1
    dataframe["calendar_year"] = dataframe["year"].str.split("/").str[1].astype(int) + 2000
    dataframe.loc[dataframe["calendar_month"].isin([11, 12]), "calendar_year"] = dataframe["calendar_year"] - 1

    return dataframe