import pandas as pd

def check_empty_dummies(X : pd.DataFrame, year : str) -> list[str]:
    """ Checks if any vars are completely 0 - want to prevent dummies being used that have zero coef"""
    no_obs = []
    num_zeroes = (X[X== 0].count()/len(X)).sort_values(ascending=False) 
    for idx, value in enumerate(num_zeroes):
        if value == 1.0:
            no_obs.append(num_zeroes.index[idx])
    if len(no_obs) > 0:
        print(f'Year: {year}. Columns found to have zero observations:')
        print(no_obs)
        return no_obs
    else:
        print(f'Year: {year}. All columns have observations')
        return []