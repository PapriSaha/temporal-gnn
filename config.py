import os


SEED = 42
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('NACC_DATA_DIR', os.path.join(BASE_DIR, 'data'))
OUT_DIR = os.environ.get('OUTPUT_DIR', os.path.join(BASE_DIR, 'results'))

CLASS_NAMES = ['CN', 'MCI', 'Dementia']
CLASS_COLORS = {'CN': '#2ecc71', 'MCI': '#f39c12', 'Dementia': '#e74c3c'}

CONFIG = {
    'seed': SEED,
    'project_name': 'CogGODE-X',
    'data_dir': DATA_DIR,
    'out_dir': OUT_DIR,
    'hidden_dim': 128,
    'n_classes': 3,
    'n_concepts': 8,
    'gat_heads': 4,
    'dropout': 0.3,
    'ode_method': 'dopri5',
    'ode_rtol': 1e-3,
    'ode_atol': 1e-4,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 100,
    'patience': 15,
    'grad_clip': 1.0,
    'lambda_ce': 1.0,
    'lambda_risk': 0.5,
    'lambda_ode': 0.01,
    'lambda_concept': 0.3,
    'n_folds': 5,
    'test_size': 0.15,
    'knn_neighbors': 15,
    'bootstrap_n': 1000,
    'n_cog_feats': 13,
    'n_func_feats': 6,
    'n_risk_feats': 14,
}
