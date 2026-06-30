data_dict = {
    # Demographic Features
    'Gend3': {
        'type' : 'categorical',
        'demographic' : True,
        'dummy_encode' : True,
        'target' : False,
        'graph_label' : 'Gender',
        'value_labels' :  {1.0: 'Male', 2.0: 'Female', 3.0: 'Other'}
        },
    'Disab3' : {
        'type' : 'categorical',
        'demographic' : True,
        'dummy_encode' : True, 
        'target' : False,
        'graph_label' : 'Disability Status',
        'value_labels' : {1.0: 'Limiting disability', 2.0: 'Non-limiting disability', 3.0: 'No disability'}
        },
    'Age9' : {
        'type' : 'ordinal',
        'demographic' : True,
        'dummy_encode' : True,
        'target' : False,
        'graph_label' : 'Age',
        'value_labels' : {1.0: '14-15', 2.0: '16-24', 3.0: '25-34', 4.0: '35-44', 5.0: '45-54', 6.0: '55-64', 7.0: '65-74', 8.0: '75-84', 9.0: '85+'}
        },
    'Eth7' : {
        'type' : 'categorical',
        'demographic' : True,
        'dummy_encode' : True,
        'target' : False,
        'graph_label' : 'Ethnicity',
        'value_labels' : {1.0: 'White British', 2.0: 'White Other', 3.0: 'Asian (excl. Chinese)', 4.0: 'Black', 5.0: 'Chinese', 6.0: 'Mixed', 7.0: 'Other ethnic group'}
        },
    'NSSEC5' : {
        'type' : 'ordinal',
        'demographic' : True,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'NSSEC',
        'value_labels' : {1.0: 'NS SEC 1-2: Higher social groups', 2.0: 'NS SEC 3-5: Middle social groups', 3.0: 'NS SEC 6-8: Lower social groups', 4.0: 'NS SEC 9: Students and other / unclassified'} 
        },

    'MEMS7_ALL' : {
        'type' : 'continuous',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : True,
        'graph_label' : 'Minutes of Moderate Activity Per Week',
        'value_labels' : None
        },
    'LOG_MEMS7_ALL' : {
        'type' : 'continuous',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : True,
        'graph_label' : 'Log Minutes of Moderate Activity Per Week',
        'value_labels' : None
        },
    'active' : {
        'type' : 'categorical',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : True,
        'graph_label' : 'Active Status',
        'value_labels' : {0 : 'Not Active', 1 : 'Active' },
        },

    # Explanatory Variables
    'Educ6' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Highest Qualification',
        'value_labels' : {1.0: 'Level 4 or above', 2.0: 'Level 3 and equivalents', 3.0: 'Level 2 and equivalents', 4.0: 'Level 1 and below', 5.0: 'Another type of qualification', 6.0: 'No qualifications'}
        },

    'IMD10' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : True,
        'target' : False,
        'graph_label' : 'IMD',
        'value_labels' : {1.0: 'Most deprived decile', 2.0: 'Second most deprived decile', 3.0: 'Third most deprived decile', 4.0: 'Fourth most deprived decile', 5.0: 'Fifth most deprived decile', 6.0: 'Fifth least deprived decile', 7.0: 'Fourth least deprived decile', 8.0: 'Third least deprived decile', 9.0: 'Second least deprived decile', 10.0: 'Least deprived decile'}
        },
    'nchild' : {
        'type' : 'continuous',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Num Children in Household',
        'value_labels' : None
        },
    'nadult' : {
        'type' : 'continuous',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Num Adults in Household',
        'value_labels' : None
        },
    'WorkStat10' : {
        'type' : 'categorical',
        'demographic' : False,
        'dummy_encode' : True,
        'target' : False,
        'graph_label' : 'Employment Status',
        'value_labels' : {1.0: 'Working full time', 2.0: 'Working part time', 3.0: 'Unemployed < 12mths', 4.0: 'Unemployed >12 mths', 5.0: 'Not working-retired', 6.0: 'Not working-looking after house/children', 7.0: 'Not working-long term sick or disabled', 8.0: 'Student full time', 9.0: 'Student part time', 10.0: 'Other'}
        },
    'VolAny' : {
        'type' : 'categorical',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Has Volunteered',
        'value_labels' : {0 : 'Not volunteered' , 1 : 'Volunteered'}
        },
        # Self-Determination Theory - intrinsic vs extrinsic motivation
    'Motiva_POP': {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Exercise enjoyable and satisfying',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'}
          },
    'motivd_POP' : { # showed massuive varaince on coefficient plot
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Exercise to prevent dissapoint from others',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'} 
        },
        # Living Arrangements
    'HHLiv12' : {
        'type' : 'categorical',
        'demographic' : False,
        'dummy_encode' : True,
        'target' : False,
        'graph_label' : 'Living Arrangements (12)',
        'value_labels' : {
            1.0: 'Single person living alone', 
            2.0: 'Houseshare', 
            3.0: 'Lone parent family with children',
            4.0: 'Lone parent family with adult children',
            5.0: 'Couple, no children, no other household members', 
            6.0: 'Couple with children', 7.0: 'Couple with adult children', 
            8.0: 'Multi-generational, no children in household',
            9.0: 'Multi-generational, children in household',
            10.0: 'Living with parents', 
            11.0: 'Other/complex, no children',
            12.0: 'Other/complex, children' }
    },
    # Time
    'year' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Year',
        'value_labels' : None
        }
}