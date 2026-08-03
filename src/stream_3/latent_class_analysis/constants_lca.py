cluster_cols = ["Age9",
                "Gend3",
                "Eth7",
                "Disab2_POP",
                "Educ6",
                "NSSEC5",
                "IMD10",
                "WorkStat8",
                "Child4",
                "HHLiv9",
                "Motiva_POP",
                "motivd_POP"]

value_labels = {

    "Age9": {
        0: "16–24",
        1: "25–34",
        2: "35–44",
        3: "45–54",
        4: "55–64",
        5: "65–74",
        6: "75–84",
        7: "85+"
    },

    "Gend3": {
        0: "Male",
        1: "Female",
        2: "Other"
    },

    "Eth7": {
        0: "White British",
        1: "White Other",
        2: "Asian (excl. Chinese)",
        3: "Black",
        4: "Chinese",
        5: "Mixed",
        6: "Other ethnic group"
    },

    "Disab2_POP": {
        0: "Disability",
        1: "No disability"
    },

    "Educ6": {
        0: "Level 4 or above",
        1: "Level 3 and equivalents",
        2: "Level 2 and equivalents",
        3: "Level 1 and below",
        4: "Another type of qualification",
        5: "No qualifications"
    },

    "NSSEC5": {
        0: "NS SEC 1–2: Higher social groups",
        1: "NS SEC 3–5: Middle social groups",
        2: "NS SEC 6–8: Lower social groups",
        3: "NS SEC 9: Students and other / unclassified",
        4: "Not applicable (75+)"
    },

    "IMD10": {
        0: "Most deprived decile",
        1: "Second most deprived decile",
        2: "Third most deprived decile",
        3: "Fourth most deprived decile",
        4: "Fifth most deprived decile",
        5: "Fifth least deprived decile",
        6: "Fourth least deprived decile",
        7: "Third least deprived decile",
        8: "Second least deprived decile",
        9: "Least deprived decile"
    },

    "WorkStat8": {
        0: "Working full time",
        1: "Working part time",
        2: "Unemployed",
        3: "Not working-retired",
        4: "Not working-looking after house/children",
        5: "Not working-long term sick or disabled",
        6: "Student",
        7: "Other"
    },

    "Child4": {
        0: "None",
        1: "1 child",
        2: "2 children",
        3: "3 or more children"
    },

    "HHLiv9": {
        0: "Single person living alone",
        1: "Houseshare",
        2: "Lone parent family",
        3: "Couple, no children",
        4: "Couple with children",
        5: "Couple with adult children",
        6: "Multi-generational household",
        7: "Living with parents",
        8: "Other/complex household"
    },

    "Motiva_POP": {
        0: "Strongly agree",
        1: "Agree",
        2: "Neither agree nor disagree",
        3: "Disagree",
        4: "Strongly disagree"
    },

    "motivd_POP": {
        0: "Strongly agree",
        1: "Agree",
        2: "Neither agree nor disagree",
        3: "Disagree",
        4: "Strongly disagree"
    }
}

display_names = {
    "Age9": "Age",

    "Gend3": "Gender",

    "Eth7": "Ethnicity",

    "Disab2_POP": "Disability",

    "Educ6": "Education",

    "NSSEC5": "Socio-economic Status",

    "IMD10": "Deprivation",

    "WorkStat8": "Employment Status",

    "Child4": "Children",

    "HHLiv9": "Household Composition",

    "VolAny": "Volunteering",

    "Motiva_POP": "Intrinsic Motivation",

    "motivd_POP": "Extrinsic Motivation"
}

merge_columns = [
    "year",
    "serial",
    "Class",
    "NSSEC5",
    "WorkStat8",
    "HHLiv9",
    "Motivation_PC_Q",
]

master_columns = [
    "serial",
    "year",
    "wt_final",
    "month",
    "LCA_Class",
    "LA_2023",
    "Age9",
    "Gend3",
    "Eth7",
    "Disab2_POP",
    "Educ6",
    "NSSEC5",
    "IMD10",
    "WorkStat8",
    "Child4",
    "HHLiv9",
    "Motiva_POP",
    "motivd_POP",
    "nadult",
    "nchild",
    "health", 
    "comm1",
    "anxious",
    "happy",
    "lifesat",
    "lone",
    "DVBMI",
    "FruitVegPor",
    "READYAB1_POP",
    "CULFRQ_1_9_POP",
    "VolAny",
    "VolCnt",
    "VolDur",
    "VolFrqB_Pop",
    "volint1",
    "volint2",
    "volint3",
    "volint4",
    "volint5",
    "volint6",
    "volint7",
    "MEMS7_ALL",
    "MEMS7_SPORTCOUNT_A01",
    "MEMS7_IN_SPORTCOUNT_A01",
    "MEMS7_OUT_SPORTCOUNT_A01",
    "MEMS7_FITNESS_B06",
    "MEMS7_WALKALL_C01",
    "MEMS7_CYCALL_C02",
    "MEMS7_ACTTRAV_C03",
    "MEMS7_DANCEALL_C04",
    "MEMS7_TEAMSPORT_C05",
    "MEMS7_RACKETSPORT_C06",
    "MEMS7_ADVWATERSPORT_C07",
    "MEMS7_LEISURE_C08",
    "MEMS7_COMBATTARGET_C09",
    "MEMS7_WINTER_C10",
    "MEMS7_RUNATHMULTI_C11",
    "ACT7GR_ALL",
    "ACT7GR_SPORTCOUNT_A01",
    "Number_Activities",
    "CLUB_SPORTCOUNT_A01",
    "Number_Club"]

forecast_aggregations = {
    "N": ("serial", "size"),
    "Sum_wt_final": ("wt_final", "sum"),
    "Mean_wt_final": ("wt_final", "mean"),
    "Mean_MEMS7_ALL": ("MEMS7_ALL", "mean"),
    "Median_MEMS7_ALL": ("MEMS7_ALL", "median"),
    "SD_MEMS7_ALL": ("MEMS7_ALL", "std"),
    "Mean_MEMS7_SPORTCOUNT_A01": ("MEMS7_SPORTCOUNT_A01", "mean"),
    "Median_MEMS7_SPORTCOUNT_A01": ("MEMS7_SPORTCOUNT_A01", "median"),
    "Mean_MEMS7_IN_SPORTCOUNT_A01": ("MEMS7_IN_SPORTCOUNT_A01", "mean"),
    "Median_MEMS7_IN_SPORTCOUNT_A01": ("MEMS7_IN_SPORTCOUNT_A01", "median"),
    "Mean_MEMS7_OUT_SPORTCOUNT_A01": ("MEMS7_OUT_SPORTCOUNT_A01", "mean"),
    "Median_MEMS7_OUT_SPORTCOUNT_A01": ("MEMS7_OUT_SPORTCOUNT_A01", "median"),
    "Mean_MEMS7_FITNESS_B06": ("MEMS7_FITNESS_B06", "mean"),
    "Median_MEMS7_FITNESS_B06": ("MEMS7_FITNESS_B06", "median"),
    "Mean_MEMS7_WALKALL_C01": ("MEMS7_WALKALL_C01", "mean"),
    "Median_MEMS7_WALKALL_C01": ("MEMS7_WALKALL_C01", "median"),
    "Mean_MEMS7_CYCALL_C02": ("MEMS7_CYCALL_C02", "mean"),
    "Median_MEMS7_CYCALL_C02": ("MEMS7_CYCALL_C02", "median"),
    "Mean_MEMS7_ACTTRAV_C03": ("MEMS7_ACTTRAV_C03", "mean"),
    "Median_MEMS7_ACTTRAV_C03": ("MEMS7_ACTTRAV_C03", "median"),
    "Mean_MEMS7_DANCEALL_C04": ("MEMS7_DANCEALL_C04", "mean"),
    "Median_MEMS7_DANCEALL_C04": ("MEMS7_DANCEALL_C04", "median"),
    "Mean_MEMS7_TEAMSPORT_C05": ("MEMS7_TEAMSPORT_C05", "mean"),
    "Median_MEMS7_TEAMSPORT_C05": ("MEMS7_TEAMSPORT_C05", "median"),
    "Mean_MEMS7_RACKETSPORT_C06": ("MEMS7_RACKETSPORT_C06", "mean"),
    "Median_MEMS7_RACKETSPORT_C06": ("MEMS7_RACKETSPORT_C06", "median"),
    "Mean_MEMS7_ADVWATERSPORT_C07": ("MEMS7_ADVWATERSPORT_C07", "mean"),
    "Median_MEMS7_ADVWATERSPORT_C07": ("MEMS7_ADVWATERSPORT_C07", "median"),
    "Mean_MEMS7_LEISURE_C08": ("MEMS7_LEISURE_C08", "mean"),
    "Median_MEMS7_LEISURE_C08": ("MEMS7_LEISURE_C08", "median"),
    "Mean_MEMS7_COMBATTARGET_C09": ("MEMS7_COMBATTARGET_C09", "mean"),
    "Median_MEMS7_COMBATTARGET_C09": ("MEMS7_COMBATTARGET_C09", "median"),
    "Mean_MEMS7_WINTER_C10": ("MEMS7_WINTER_C10", "mean"),
    "Median_MEMS7_WINTER_C10": ("MEMS7_WINTER_C10", "median"),
    "Mean_MEMS7_RUNATHMULTI_C11": ("MEMS7_RUNATHMULTI_C11", "mean"),
    "Median_MEMS7_RUNATHMULTI_C11": ("MEMS7_RUNATHMULTI_C11", "median"),
    "Median_ACT7GR_ALL": ("ACT7GR_ALL", "median"),
    "Median_ACT7GR_SPORTCOUNT_A01": ("ACT7GR_SPORTCOUNT_A01", "median"),
    "Mean_Number_Activities": ("Number_Activities", "mean"),
    "Median_Number_Activities": ("Number_Activities", "median"),
    "Mean_CLUB_SPORTCOUNT_A01": ("CLUB_SPORTCOUNT_A01", "mean"),
    "Mean_Number_Club": ("Number_Club", "mean"),
    "Median_Number_Club": ("Number_Club", "median"),
    "Mean_VolAny": ("VolAny", "mean"),
    "Mean_VolCnt": ("VolCnt", "mean"),
    "Median_VolCnt": ("VolCnt", "median"),
    "Mean_VolDur": ("VolDur", "mean"),
    "Median_VolDur": ("VolDur", "median"),
    "Mean_VolFrqB_Pop": ("VolFrqB_Pop", "mean"),
    "Median_VolFrqB_Pop": ("VolFrqB_Pop", "median"),
    "Mean_volint1": ("volint1", "mean"),
    "Mean_volint2": ("volint2", "mean"),
    "Mean_volint3": ("volint3", "mean"),
    "Mean_volint4": ("volint4", "mean"),
    "Mean_volint5": ("volint5", "mean"),
    "Mean_volint6": ("volint6", "mean"),
    "Mean_volint7": ("volint7", "mean"),
    "Mean_DVBMI": ("DVBMI", "mean"),
    "Median_DVBMI": ("DVBMI", "median"),
    "Mean_FruitVegPor": ("FruitVegPor", "mean"),
    "Median_FruitVegPor": ("FruitVegPor", "median")
}

class_names = {
    0:  "Mid-life working fathers",
    1:  "Young highly educated professional women",
    2:  "Older unemployed disadvantaged adults",
    3:  "Young adults living with parents",
    4:  "Affluent older working women",
    5:  "Oldest retired adults 1",
    6:  "Mid-life professional women",
    7:  "Highly educated professional fathers",
    8:  "Young highly educated professionals",
    9:  "Disabled mid-life adults",
    10: "Affluent older professionals",
    11: "Older middle socioeconomic workers",
    12: "Lone mothers in lower socioeconomic households",
    13: "Oldest retired adults 2",
    14: "Established professional adults",
    15: "Older transitioning retirees",
    16: "Young students living with parents",
    17: "University students in shared housing",
    18: "Young Asian professionals",
    19: "Lower socioeconomic family mothers",
    20: "Highly educated young professionals",
    21: "Professional mothers",
    22: "Working family mothers",
    23: "Retired older adults",
    24: "Affluent retired older adults"
}

included_years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]

forecast_years = [2023, 2024, 2025, 2026, 2027]