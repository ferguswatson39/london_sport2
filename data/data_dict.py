
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
        'graph_label' : 'Minutes of Moderate Activity Per Week'
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
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'IMD',
        'value_labels' : {1.0: 'Most deprived decile', 2.0: 'Second most deprived decile', 3.0: 'Third most deprived decile', 4.0: 'Fourth most deprived decile', 5.0: 'Fifth most deprived decile', 6.0: 'Fifth least deprived decile', 7.0: 'Fourth least deprived decile', 8.0: 'Third least deprived decile', 9.0: 'Second least deprived decile', 10.0: 'Least deprived decile'}
        },
    'nchild' : {
        'type' : 'continuous',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Num Children in Household'
        },
    'nadult' : {
        'type' : 'continuous',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Num Adults in Household'
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
        # Motivations
    'motivex2a' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Exercise for Fitness and Health',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'}
        },
    'motivex2b' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Exercise to relax and worry less',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'},
        },
    
    'motivex2c' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Exercise socially for fun with friends',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'}
        },
    'motivex2d' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Exercise to challenge myself',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'}
        },
    'Motiva_POP': {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Exercise enjoyable and satisfying',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'}
          },
    'motivb_POP' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Regular exercise is important',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'}
        },
    'motivc_POP' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Guilt when no exercise',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'}
        },
    'motivd_POP' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Exercise to prevent guilt from others',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'} 
        },
    'motive_POP' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Exercise is pointless',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'}
        },
        # Different Motivations
    'indev' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Can achieve goals',
        'value_labels' : {1.0: 'Strongly disagree', 2.0: 'Disagree', 3.0: 'Neither agree nor disagree', 4.0: 'Agree', 5.0: 'Strongly agree'}
        },
    'indevtry' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Persistence (keep trying)',
        'value_labels' : {1.0: 'Strongly disagree', 2.0: 'Disagree', 3.0: 'Neither agree nor disagree', 4.0: 'Agree', 5.0: 'Strongly agree'}
        },
        # Social Cohesion
    'inclus_a' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Exercise areas inclusive and welcoming',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'}
        },
    'inclus_b' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'See similar people when exercise',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'}
        },
    'inclus_c' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Safe exercise places (public)',
        'value_labels' : {1.0: 'Strongly agree', 2.0: 'Agree', 3.0: 'Neither agree nor disagree', 4.0: 'Disagree', 5.0: 'Strongly disagree'}
        },
    'comm1' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Trust local people',
        'value_labels' : {1.0: 'Strongly disagree', 2.0: 'Disagree', 3.0: 'Neither agree nor disagree', 4.0: 'Agree', 5.0: 'Strongly agree'}
        },
    'comm2' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Local people from different backgrounds get on well',
        'value_labels' : {-2.0: 'People in this area are all of the same background', -1.0: 'There are too few people in the local area', 1.0: 'Definitely disagree', 2.0: 'Tend to disagree', 3.0: 'Tend to agree', 4.0: 'Definitely agree'}
        },
        # Life Emotions
    'anxious' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Anxious yesterday',
        'value_labels' : {0.0: '0 Not at all anxious', 1.0: '1', 2.0: '2', 3.0: '3', 4.0: '4', 5.0: '5', 6.0: '6', 7.0: '7', 8.0: '8', 9.0: '9', 10.0: '10 Completely anxious'}
        },
    'happy' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Happy yesterday',
        'value_labels' : {0.0: '0 Not at all happy', 1.0: '1', 2.0: '2', 3.0: '3', 4.0: '4', 5.0: '5', 6.0: '6', 7.0: '7', 8.0: '8', 9.0: '9', 10.0: '10 Completely happy'}
        },
    'lifesat' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Life Satisfaction',
        'value_labels' : {0.0: '0 Not at all satisfied', 1.0: '1', 2.0: '2', 3.0: '3', 4.0: '4', 5.0: '5', 6.0: '6', 7.0: '7', 8.0: '8', 9.0: '9', 10.0: '10 Completely satisfied'}
        },
    'lone' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Feeling lonely',
        'value_labels' : {1.0: 'Often / always', 2.0: 'Some of the time', 3.0: 'Occasionally', 4.0: 'Hardly ever', 5.0: 'Never'}
        },
    'worthw' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Extent life worthwhile',
        'value_labels' : {0.0: '0 Not at all worthwhile', 1.0: '1', 2.0: '2', 3.0: '3', 4.0: '4', 5.0: '5', 6.0: '6', 7.0: '7', 8.0: '8', 9.0: '9', 10.0: '10 Completely worthwhile'}
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
    'HHLiv23' : {'type' : 'categorical',
        'demographic' : False,
        'dummy_encode' : True,
        'target' : False,
        'graph_label' : 'Living Arrangements (23)',
        'value_labels' : {
            1.0: 'Single person living alone',
            2.0: 'Single person living with adult non-relatives',
            3.0: 'Lone parent with children under 16 in the household and with children (regardless of age) resident full time',
            4.0: 'Lone parent with children under 16 in the household and with children (regardless of age) only resident part time',
            5.0: 'Lone parent with children 16 and over in the household (and no children under 16)',
            6.0: 'Couple household (no other household members) and no children ever', 
            7.0: 'Couple household (with lodger(s)) and no children ever',
            8.0: 'Couple household (no other household members), children have left home',
            9.0: 'Couple household (with lodger(s)), children have left home', 
            10.0: 'Couple with children under 16 in the household and with children (regardless of age) resident full time',
            11.0: 'Couple with children under 16 in the household and with children (regardless of age) only resident part time',
            12.0: 'Couple with children 16 and over in the household (and no children under 16)',
            13.0: 'Single adult living in family with three or more generations (with children under 16 in HH)', 
            14.0: 'Single adult living in family with three or more generations (with no children under 16 in HH)',
            15.0: 'Cohabiting couple living in family with three or more generations (with children under 16 in HH)', 
            16.0: 'Cohabiting couple living in family with three or more generations (with no children under 16 in HH)',
            17.0: 'Single adult living with parent(s)', 
            18.0: 'Cohabiting couple living with parents', 
            19.0: 'Single adult living with siblings', 
            20.0: 'Lone parent living in other/complex family',
            21.0: 'Single adult living in other/complex family', 
            22.0: 'Cohabiting couple (with children under 16) living in other/complex family', 
            23.0: 'Cohabiting couple (without children under 16) living in other/complex family'}
    
    },
    'HHLiv5' : {'type' : 'categorical',
        'demographic' : False,
        'dummy_encode' : True,
        'target' : False,
        'graph_label' : 'Living Arrangements (5)',
        'value_labels' : {
            1.0: 'Single person', 
            2.0: 'Lone parent',
            3.0: 'Couple', 
            4.0: 'Multi-generational household', 
            5.0: 'Other/ complex household'}
    },
    # Time
    'year' : {
        'type' : 'ordinal',
        'demographic' : False,
        'dummy_encode' : False,
        'target' : False,
        'graph_label' : 'Year',
        'value_labels' : {
            '2015/16' : '2015/16',
            '2016/17' : '2016/17',
            '2017/18' : '2017/18',
            '2018/19' : '2018/19',
            '2019/20' : '2019/20',
            '2020/21' : '2020/21',
            '2021/22' : '2021/22',
            '2022/23' : '2022/23'
            }
        }
}

