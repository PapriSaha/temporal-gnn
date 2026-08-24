import os

import pandas as pd
import numpy as np


class NACCDataLoader:

    ID_COLS = ['NACCID', 'NACCADC', 'PACKET', 'FORMVER', 'VISITYR', 'VISITMO', 'NACCVNUM']
    DEMO_COLS = ['BIRTHYR', 'NACCSEX', 'EDUC', 'RACE', 'NACCHISP', 'MARISTAT', 'INDEPEND']
    COMORBID_COLS = ['DIABETES', 'HYPERTEN', 'HYPERCHO', 'CVHATT', 'CVAFIB',
                     'CBSTROKE', 'CBTIA', 'TOBAC100', 'NACCBMI']
    GENETICS_COLS = ['NACCNE4S']
    GLOBAL_COG_RAW = ['NACCMMSE', 'NACCMOCA']
    MEMORY_RAW = ['LOGIMEM', 'CRAFTVRS', 'CRAFTDVR']
    NAMING_RAW = ['BOSTON', 'MINTTOTS']
    TRAIL_COLS = ['TRAILA', 'TRAILB']
    DIGIT_COLS = ['DIGIF', 'DIGIB', 'DIGFORCT', 'DIGBACCT']
    FLUENCY_COLS = ['ANIMALS', 'VEG']
    CDR_COLS = ['CDRSUM']
    FAQ_ITEM_COLS = ['BILLS', 'TAXES', 'SHOPPING', 'STOVE', 'TRAVEL']
    NPIQ_SEVERITY_COLS = ['ANXSEV', 'APASEV', 'AGITSEV', 'DELSEV', 'HALLSEV', 'IRRSEV', 'MOTSEV']
    GDS_COLS = ['NACCGDS']
    TARGET_COL = 'NACCUDSD'

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.all_feature_cols = (self.DEMO_COLS + self.COMORBID_COLS + self.GENETICS_COLS +
                                 self.GLOBAL_COG_RAW + self.MEMORY_RAW + self.NAMING_RAW +
                                 self.TRAIL_COLS + self.DIGIT_COLS + self.FLUENCY_COLS +
                                 self.CDR_COLS + self.FAQ_ITEM_COLS + self.NPIQ_SEVERITY_COLS +
                                 self.GDS_COLS)
        self.consort = {}

    def load_raw(self):
        path = os.path.join(self.data_dir, 'synthetic_dataset.csv')
        df = pd.read_csv(path, low_memory=False, dtype={'NACCID': str})
        df['VISITDATE'] = pd.to_datetime(df['VISITDATE'], errors='coerce')
        df['VISITYR'] = df['VISITDATE'].dt.year
        df['VISITMO'] = df['VISITDATE'].dt.month
        df['AGE_AT_VISIT'] = df['VISITYR'] - df['BIRTHYR']
        if 'BIRTHMO' in df.columns:
            df.loc[df['VISITMO'] < df['BIRTHMO'], 'AGE_AT_VISIT'] -= 1
        return df

    def apply_filters(self, df):
        self.consort['total_visits'] = len(df)
        self.consort['total_participants'] = df['NACCID'].nunique()

        df_sel = df[df[self.TARGET_COL].isin([1, 3, 4])].copy()
        df_sel['LABEL'] = df_sel[self.TARGET_COL].map({1: 0, 3: 1, 4: 2})
        self.consort['after_dx_filter_visits'] = len(df_sel)
        self.consort['after_dx_filter_participants'] = df_sel['NACCID'].nunique()

        df_sel = df_sel[df_sel['AGE_AT_VISIT'] >= 50]
        self.consort['after_age_filter_visits'] = len(df_sel)
        self.consort['after_age_filter_participants'] = df_sel['NACCID'].nunique()

        visit_counts = df_sel.groupby('NACCID')['NACCVNUM'].nunique()
        multi_visit = visit_counts[visit_counts >= 2].index
        df_sel = df_sel[df_sel['NACCID'].isin(multi_visit)]
        self.consort['after_visit_filter_visits'] = len(df_sel)
        self.consort['after_visit_filter_participants'] = df_sel['NACCID'].nunique()

        return df_sel.sort_values(['NACCID', 'NACCVNUM']).reset_index(drop=True)

    def get_cohort(self):
        df = self.load_raw()
        return self.apply_filters(df)

    def get_consort_counts(self):
        return self.consort
