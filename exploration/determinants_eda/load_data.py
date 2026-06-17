import pandas as pd
import numpy as np
import pyreadstat

def get_2022_data():
    DATAPATH = r"C:/Masters/London Sport/9288_ActiveLifeSurvey_2022_2023/UKDA-9288-spss/spss/spss28/active_lives_survey_nov_22-23_data_year_8_shared_20250103.sav"
    COLS = [
    'wt_final', 'wt_time', 'Reg9', 'xStrata', 'Relig7', 'LondInOut', 'LA_2023',
    'Age9', 'nchild', 'nadult', 'Disab3', 'Gend3', 'Educ6', 'Orient4',

    'disty1_POP', 'disty2_POP', 'disty3_POP', 'disty4_POP', 'disty5_POP',
    'disty6_POP', 'disty7_POP', 'disty8_POP', 'disty9_POP', 'disty10_POP',
    'disty11_POP', 'disty12_POP', 'disty13_POP',

    'Eth2', 'Eth7', 'health', 'NSSEC4', 'NSSEC5', 'NSSEC8',
    'IMD10', 'IMD10_GR2', 'IMD10_GR3', 'IMD10_GR5', 'IMD4',
    'WorkStat10', 'WorkStat5', 'WorkStat7', 'Equalities_Metric_2024_GR2',

    'BMIG', 'DVBMI', 'DVHeightM', 'DVWeightKG', 'fruitveg',
    'MEMS7_ALL', 'MEMS7GR_ALL', 'MEMS7_SPORTCOUNT_A01',
    'MEMS7_OUT_SPORTCOUNT_A01', 'MEMS7_IN_SPORTCOUNT_A01',
    'DURATION_SPORTCOUNT_A01',

    'VolAny', 'VolCnt', 'VolFrqB_Pop',
    'volint1', 'volint2', 'volint3', 'volint4', 'volint5', 'volint6', 'volint7',

    'Motiva_POP', 'motivb_POP', 'motivc_POP', 'motivd_POP', 'motive_POP',
    'motivex2a', 'motivex2b', 'motivex2c', 'motivex2d',

    'inclus_a', 'inclus_b', 'inclus_c',
    'anxious', 'comm1', 'comm2', 'happy',
    'indev', 'indevtry', 'lifesat', 'lone', 'worthw', 'workactlvl',

    'WHOWITHA_SPORTCOUNT_A01', 'WHOWITHB_SPORTCOUNT_A01',
    'WHOWITHC_SPORTCOUNT_A01', 'WHOWITHD_SPORTCOUNT_A01',

    'limfreti1', 'limfreti2', 'limfreti3', 'limfreti4',
    'limfreti5', 'limfreti6', 'limfreti7', 'limfreti8',

    'MONTHS_12_FOOTBALL_F01', 'MONTHS_12_CRICKET_F02',
    'MONTHS_12_RUGBYUNION_F03', 'MONTHS_12_RUGBYLEAGUE_F04',
    'MONTHS_12_WHEELCHRUGBY_F05', 'MONTHS_12_NETBALL_F06',
    'MONTHS_12_BASKETBALL_F07', 'MONTHS_12_WHEELCHBASKETBALL_F08',
    'MONTHS_12_HOCKEY_F09', 'MONTHS_12_VOLLEYBALL_F10',
    'MONTHS_12_ROUNDERS_F11', 'MONTHS_12_DODGEBALL_F12',
    'MONTHS_12_BASESOFTBALL_F13', 'MONTHS_12_LACROSSE_F14',
    'MONTHS_12_GOALBALL_F15', 'MONTHS_12_HANDBALL_F16',

    'CULFRQ_1_9_POP', 'CULFRQ_1_9_POP2', 'CULMTH_1_9_POP'
    ]

    df, meta = pyreadstat.read_sav(DATAPATH, usecols = COLS)
    df = df[df['LondInOut'].notna()]
    for c in COLS:
        if c not in df.columns:
            raise ValueError(f'Column: {c} not in df')
    print('Data Collected...')
    print(f'Data Shape: {df.shape}')
    return df, meta

