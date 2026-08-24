import os
import pickle

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class Preprocessor:

    NACC_MISSING_CODES = [-4, -9, 88, 95, 96, 97, 98, 99, 888, 999, 8888, 9999]

    COG_RAW_V12 = ['NACCMMSE', 'LOGIMEM', 'BOSTON', 'DIGIF', 'DIGIB']
    COG_RAW_V34 = ['MOCATOTS', 'CRAFTVRS', 'CRAFTDVR', 'MINTTOTS', 'DIGFORCT', 'DIGBACCT']
    SHARED_COG = ['TRAILA', 'TRAILB', 'ANIMALS', 'VEG']

    COG_FEATS = ['GLOBAL_COG_RAW_SCORE', 'MEMORY_RAW_SCORE', 'NAMING_RAW_SCORE',
                 'TRAILA', 'TRAILB', 'DIGIF_SCORE', 'DIGIB_SCORE',
                 'ANIMALS', 'VEG', 'CDRSUM', 'NACCGDS',
                 'GLOBAL_COG_RAW_SCORE_DELTA', 'GLOBAL_COG_RAW_SCORE_RATE']
    FUNC_FEATS = ['FAQ_TOTAL', 'INDEPEND', 'FAQ_TOTAL_DELTA',
                  'FAQ_TOTAL_RATE', 'FOLLOWUP_YEARS', 'VISIT_INTERVAL']
    RISK_FEATS = ['NACCSEX', 'AGE_AT_VISIT', 'EDUC', 'RACE_WHITE', 'RACE_BLACK',
                  'RACE_OTHER', 'NACCNE4S', 'NACCBMI', 'DIABETES', 'HYPERTEN',
                  'HYPERCHO', 'CVAFIB', 'CBSTROKE', 'TOBAC100']

    def __init__(self, config):
        self.config = config
        self.imputer = None
        self.scalers = {}
        self.missing_report = {}

    def replace_missing_codes(self, df, cols):
        for col in cols:
            if col in df.columns:
                df[col] = df[col].replace(self.NACC_MISSING_CODES, np.nan)
        return df

    def harmonize_uds_versions(self, df):
        df['FORMVER_NUM'] = pd.to_numeric(df['FORMVER'], errors='coerce').fillna(3).astype(int)
        is_v12 = df['FORMVER_NUM'].isin([1, 2])
        is_v34 = ~is_v12

        df['GLOBAL_COG_RAW_SCORE'] = np.nan
        df.loc[is_v12, 'GLOBAL_COG_RAW_SCORE'] = df.loc[is_v12, 'NACCMMSE']
        df.loc[is_v34, 'GLOBAL_COG_RAW_SCORE'] = df.loc[is_v34, 'MOCATOTS'] if 'MOCATOTS' in df.columns else df.loc[is_v34, 'NACCMOCA']

        df['MEMORY_RAW_SCORE'] = np.nan
        df.loc[is_v12, 'MEMORY_RAW_SCORE'] = df.loc[is_v12, 'LOGIMEM']
        if 'CRAFTVRS' in df.columns:
            df.loc[is_v34, 'MEMORY_RAW_SCORE'] = df.loc[is_v34, ['CRAFTVRS', 'CRAFTDVR']].mean(axis=1)

        df['NAMING_RAW_SCORE'] = np.nan
        df.loc[is_v12, 'NAMING_RAW_SCORE'] = df.loc[is_v12, 'BOSTON']
        if 'MINTTOTS' in df.columns:
            df.loc[is_v34, 'NAMING_RAW_SCORE'] = df.loc[is_v34, 'MINTTOTS']

        df['DIGIF_SCORE'] = np.nan
        df.loc[is_v12, 'DIGIF_SCORE'] = df.loc[is_v12, 'DIGIF']
        if 'DIGFORCT' in df.columns:
            df.loc[is_v34, 'DIGIF_SCORE'] = df.loc[is_v34, 'DIGFORCT']

        df['DIGIB_SCORE'] = np.nan
        df.loc[is_v12, 'DIGIB_SCORE'] = df.loc[is_v12, 'DIGIB']
        if 'DIGBACCT' in df.columns:
            df.loc[is_v34, 'DIGIB_SCORE'] = df.loc[is_v34, 'DIGBACCT']

        if 'RACE' in df.columns:
            df['RACE_WHITE'] = (df['RACE'] == 1).astype(int)
            df['RACE_BLACK'] = (df['RACE'] == 2).astype(int)
            df['RACE_OTHER'] = (~df['RACE'].isin([1, 2])).astype(int)

        return df

    def compute_derived_features(self, df):
        df['VISIT_DATE_FRAC'] = df['VISITYR'] + (df['VISITMO'] - 1) / 12.0

        faq_all_10 = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
                      'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
        faq_present = [c for c in faq_all_10 if c in df.columns]
        df['FAQ_TOTAL'] = df[faq_present].sum(axis=1, min_count=1)
        if len(faq_present) < 10:
            df['FAQ_TOTAL'] = df['FAQ_TOTAL'] * (10 / len(faq_present))

        df['FOLLOWUP_YEARS'] = df.groupby('NACCID')['VISIT_DATE_FRAC'].transform(
            lambda x: x - x.iloc[0])
        df['VISIT_INTERVAL'] = df.groupby('NACCID')['VISIT_DATE_FRAC'].diff()

        df['FAQ_TOTAL_DELTA'] = df.groupby('NACCID')['FAQ_TOTAL'].diff()
        df['FAQ_TOTAL_RATE'] = df['FAQ_TOTAL_DELTA'] / (df['VISIT_INTERVAL'] + 1e-8)

        df['GLOBAL_COG_RAW_SCORE_DELTA'] = df.groupby('NACCID')['GLOBAL_COG_RAW_SCORE'].diff()
        df['GLOBAL_COG_RAW_SCORE_RATE'] = df['GLOBAL_COG_RAW_SCORE_DELTA'] / (df['VISIT_INTERVAL'] + 1e-8)

        last_label = df.groupby('NACCID')['LABEL'].transform('last')
        first_label = df.groupby('NACCID')['LABEL'].transform('first')
        df['PROGRESSES'] = (last_label > first_label).astype(int)

        return df

    def handle_missing(self, df, feature_cols, method='locf_median'):
        if method == 'locf_median':
            for col in feature_cols:
                if col in df.columns:
                    df[col] = df.groupby('NACCID')[col].ffill()
            medians = df[feature_cols].median()
            df[feature_cols] = df[feature_cols].fillna(medians)
            return df, medians
        elif method == 'mice':
            imp = IterativeImputer(max_iter=10, random_state=42)
            present = [c for c in feature_cols if c in df.columns]
            df[present] = imp.fit_transform(df[present])
            return df, imp
        return df, None

    def normalize(self, df, continuous_cols, binary_cols):
        for col in continuous_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                    self.scalers[col] = (mean, std)
        return df

    def get_feature_definitions(self):
        concept_names = ['Global Cognition', 'Memory', 'Executive Function',
                         'Language', 'Functional Status', 'Behavioral Symptoms',
                         'Vascular Risk', 'Genetic Risk']
        return {
            'COG_FEATS': self.COG_FEATS,
            'FUNC_FEATS': self.FUNC_FEATS,
            'RISK_FEATS': self.RISK_FEATS,
            'concept_names': concept_names,
        }

    def run(self, df, out_dir=None):
        all_source = (self.COG_RAW_V12 + self.COG_RAW_V34 + self.SHARED_COG +
                       ['CDRSUM', 'NACCGDS'] + ['NACCSEX', 'AGE_AT_VISIT', 'EDUC', 'RACE',
                       'NACCNE4S', 'NACCBMI'] + ['DIABETES', 'HYPERTEN', 'HYPERCHO',
                       'CVAFIB', 'CBSTROKE', 'CBTIA', 'TOBAC100'] +
                       ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
                        'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL'] +
                       ['ANXSEV', 'APASEV', 'AGITSEV', 'DELSEV', 'HALLSEV', 'IRRSEV', 'MOTSEV'] +
                       ['INDEPEND', 'FORMVER'])

        df = self.replace_missing_codes(df, all_source)
        df = self.harmonize_uds_versions(df)
        df = self.compute_derived_features(df)

        all_feats = self.COG_FEATS + self.FUNC_FEATS + self.RISK_FEATS
        continuous = [c for c in all_feats if c in df.columns and df[c].nunique() > 10]
        binary = [c for c in all_feats if c in df.columns and df[c].nunique() <= 10]

        df, medians = self.handle_missing(df, [c for c in all_feats if c in df.columns])
        df = self.normalize(df, continuous, binary)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(os.path.join(out_dir, 'preprocessed_cohort.csv'), index=False)
            feat_defs = self.get_feature_definitions()
            with open(os.path.join(out_dir, 'feature_definitions.pkl'), 'wb') as f:
                pickle.dump(feat_defs, f)

        return df
