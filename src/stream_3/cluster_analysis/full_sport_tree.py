import pandas as pd
class FullSportTree:
    def __init__(self, df : pd.DataFrame, measure: str):
        self.df = df
        self.measure = measure
        self.tree = {
          'WALKALL_C01' : {
               'description' : 'Walking All',
               'num_non_zero' : self.get_non_zero('WALKALL_C01'),
               'children' : {
                    'WALKLEISURE_B01' : {
                         'description' : 'Walking for leisure',
                         'num_non_zero' : self.get_non_zero('WALKLEISURE_B01'),
                         'children' : {}
                    }},
                    'WALKTRAV_B02' : {
                         'description' : 'Walking for travel',
                         'num_non_zero' : self.get_non_zero('WALKTRAV_B02'),
                         'children' : {}
                    }},
          'CYCALL_C02': {
               'description' : 'Cycling All',
               'num_non_zero' : self.get_non_zero('CYCALL_C02'),
               'children' : {

               'CYCLEISSPORT_B03' : {
                    'description' : 'Cycling for leisure and sport',
                    'num_non_zero' : self.get_non_zero('CYCLEISSPORT_B03'),
                    'children' : {

                    'CYCLEISURE_N01' : {
                              'description' : 'Cycling for leisure',
                              'num_non_zero' : self.get_non_zero('CYCLEISURE_N01'),
                              'children' : {}
                         },

                         'CYCMOUNTAIN_N02' : {
                              'description' : 'Mountain biking',
                              'num_non_zero' : self.get_non_zero('CYCMOUNTAIN_N02'),
                              'children' : {}
                         },

                         'CYCBMX_N03' : {
                              'description' : 'BMX',
                              'num_non_zero' : self.get_non_zero('CYCBMX_N03'),
                              'children' : {}                                
                         },

                         'CYCROAD_N04' : {
                              'description' : 'Road cycling',
                              'num_non_zero' : self.get_non_zero('CYCROAD_N04'),
                              'children' : {}
                         },

                         'CYCTRACK_N05' : {
                              'description' : 'Track cycling',
                              'num_non_zero' : self.get_non_zero('CYCTRACK_N05'),
                              'children' : {}
                         },

                         'CYCCROSS_N06' : {
                              'description' : 'Cyclo-Cross',
                              'num_non_zero' : self.get_non_zero('CYCCROSS_N06'),
                              'children' : {}
                         }},

               'CYCTRAV_B04' : {
                    'description' : 'Cycling for travel',
                    'num_non_zero' : self.get_non_zero('CYCTRAV_B04'),
                    'children' : {}
                    },
          
          'ACTTRAV_C03' : {
               'description' : 'Active travel (walking or cycling)',
               'num_non_zero' : self.get_non_zero('ACTTRAV_C03'),
               'children' : {} # technically children here should be 

          }

                           
               }
          },
          },
                        
                        
                        
                        
                        }
    def get_non_zero(self, var_name) -> int:
            """Returns the proportion of non zero values"""
            full_name = f'{self.measure}_{var_name}'
            num = (self.df[full_name] != 0).sum()
            return int(num)/len(self.df)
    def get_tree(self):
         return self.tree