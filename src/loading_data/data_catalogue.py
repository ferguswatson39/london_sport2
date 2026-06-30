from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from loading_data.data_dict import data_dict
import pandas as pd

class DataCatalogue:
    def __init__(self, data_dict : dict = data_dict):
        self.data_dict = data_dict

    def get_data_dict(self) -> dict:
        return self.data_dict

    def get_to_be_encoded_vars(self) -> list:
        vars = []
        for key, value in self.data_dict.items():
            if self.data_dict[key]['dummy_encode'] == True:
                vars.append(key)
        return vars
    
    def get_demographic_vars(self) -> list:
        vars = []
        for key, value in self.data_dict.items():
            if self.data_dict[key]['demographic'] == True:
                vars.append(key)
        return vars
    
    def get_vars(self) -> list: 
        vars = [key for key, value in self.data_dict.items()]
        return vars
    
    def get_continuous_vars(self, with_targets : bool) -> list[str]:
        if with_targets:
            vars = [key for key, value in self.data_dict.items() if self.data_dict[key]['type'] == 'continuous']
            return vars
        vars = [key for key, value in self.data_dict.items() if self.data_dict[key]['type'] == 'continuous' and self.data_dict[key]['target'] == False]
        return vars
    
    def get_vars_to_be_scaled(self):
        vars = [key for key, value in self.data_dict.items() if self.data_dict[key]['type'] == 'continuous' or self.data_dict[key]['type'] == 'ordinal' and key != 'Age9' ]
        return vars

    def check_var_health(self, var : str, df : pd.DataFrame, health : int = 80000):
        try : 
            if len(df[var].dropna()) < health:
                return False
            return True
        except KeyError:
            print(f'{var} not in df')

    def get_healthy_vars(self, df : pd.DataFrame) -> list[str]:
        healthy = []
        for key, value in self.data_dict.items():
            if self.check_var_health(key, df):
                healthy.append(key)
        return healthy
    
    def get_target_vars(self):
        targets = []
        for key, value in self.data_dict.items():
            if self.data_dict[key]['target'] == True:
                targets.append(key)
        return targets

    

