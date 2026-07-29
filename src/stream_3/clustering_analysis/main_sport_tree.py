import pandas as pd
class MainSportTree:
    def __init__(self, df : pd.DataFrame, measure : str):
        self.df = df
        self.measure = measure
        self.tree = {
            'WALKALL_C01' : {
                'description' : 'Walking (All)',
                'num_non_zero' : self.get_non_zero('WALKALL_C01'),
                'children' : {
                    'WALKLEISURE_B01' : {
                        'description' : 'Walking (Leisure)',
                        'num_non_zero' : self.get_non_zero('WALKLEISURE_B01'),
                        'children' : {}
                    },
                    'WALKTRAV_B02' : {
                        'description' : 'Walking (Travel)',
                        'num_non_zero' : self.get_non_zero('WALKTRAV_B02'),
                        'children' : {}
                    }
                }
            },
            'CYCALL_C02': {
               'description' : 'Cycling All',
               'num_non_zero' : self.get_non_zero('CYCALL_C02'),
               'children' : {

               'CYCLEISSPORT_B03' : {
                    'description' : 'Cycling for leisure and sport',
                    'num_non_zero' : self.get_non_zero('CYCLEISSPORT_B03'),
                    'children' : {}
               },
               'CYCTRAV_B04' : {
                    'description' : 'Cycling for travel',
                    'num_non_zero' : self.get_non_zero('CYCTRAV_B04'),
                    'children' : {}
                    }
                }
            },
            'FITNESS_B06' : {
                'description' : 'Fitness',
                'num_non_zero' : self.get_non_zero('FITNESS_B06'),
                'children' : {}

            },
            'TEAMSPORT_C05' : {
                'descritpion' : 'Team Sports',
                'num_non_zero' : self.get_non_zero('TEAMSPORT_C05'),
                'children' : {}
            },
            'RACKETSPORT_C06' : {
                'description' : 'Racket Sports',
                'num_non_zero' : self.get_non_zero('RACKETSPORT_C06'),
                'children' : {}
            },
            'GOLF_L08' : {
                'description' : 'Golf',
                'num_non_zero' : self.get_non_zero('GOLF_L08'),
                'children' : {}
            },
            'GARD_B08' : {
                'description' : 'Gardening',
                'num_non_zero' : self.get_non_zero('GARD_B08'),
                'children' : {}
            },
            'LEISURE_C08' : {
                'description' : 'Leisure activities',
                'num_non_zero' : self.get_non_zero('LEISURE_C08'),
                'children' : {}
            }
        }
    def get_non_zero(self, var_name) -> int:
            """Returns the proportion of non zero values"""
            full_name = f'{self.measure}_{var_name}'
            num = (self.df[full_name] != 0).sum()
            return int(num)/len(self.df)
    def get_tree(self):
         return self.tree
