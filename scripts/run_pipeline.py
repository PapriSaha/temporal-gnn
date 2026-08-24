import argparse
import os
import sys
import pickle

# Add the parent directory containing 'config.py' to sys.path
_current_dir = os.path.dirname(__file__)
_parent_dir = os.path.dirname(_current_dir)
if os.path.exists(os.path.join(_parent_dir, 'config.py')):
    sys.path.insert(0, _parent_dir)
else:
    sys.path.insert(0, os.path.join(_parent_dir, 'src'))

from config import CONFIG, SEED
from utils import set_seed, setup_output_dirs, save_config, get_device


def main():
    parser = argparse.ArgumentParser(description='CogGODE-X Pipeline')
    parser.add_argument('--data-dir', default=CONFIG['data_dir'])
    parser.add_argument('--output-dir', default=CONFIG['out_dir'])
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n-epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-explain', action='store_true')
    args = parser.parse_args()

    CONFIG['data_dir'] = args.data_dir
    CONFIG['out_dir'] = args.output_dir
    out_dir = args.output_dir

    set_seed(args.seed)
    setup_output_dirs(out_dir)
    save_config(CONFIG, out_dir)
    device = get_device()
    print(f'Device: {device}')

    print('\n[Step 1/6] Loading data...')
    from data import NACCDataLoader
    loader = NACCDataLoader(args.data_dir)
    df = loader.get_cohort()
    consort = loader.get_consort_counts()
    print(f'  Cohort: {len(df):,} visits, {df["NACCID"].nunique():,} participants')
    df.to_csv(os.path.join(out_dir, 'raw_cohort.csv'), index=False)

    print('\n[Step 2/6] Preprocessing...')
    from preprocessing import Preprocessor
    preprocessor = Preprocessor(CONFIG)
    df = preprocessor.run(df, out_dir=out_dir)
    feat_defs = preprocessor.get_feature_definitions()
    print(f'  Features: COG={len(feat_defs["COG_FEATS"])}, '
          f'FUNC={len(feat_defs["FUNC_FEATS"])}, '
          f'RISK={len(feat_defs["RISK_FEATS"])}')

    print('\n[Step 3/6] Building graphs...')
    import torch
    from graph_construction import GraphBuilder
    builder = GraphBuilder(
        feat_defs['COG_FEATS'], feat_defs['FUNC_FEATS'],
        feat_defs['RISK_FEATS'], feat_defs['concept_names'])
    graphs, errors = builder.build_all_graphs(df)
    if errors:
        print(f'  Errors: {len(errors)} patients skipped')
    in_dims = builder.save_graphs(graphs, out_dir)
    print(f'  Graphs: {len(graphs):,}, In dims: {in_dims}')

    import numpy as np
    labels = np.array([g.y.item() for g in graphs])
    adrc_ids = np.array([g.adrc for g in graphs])

    pid_to_prog = df.groupby('NACCID')['PROGRESSES'].max().to_dict()
    prog_labels = np.array([pid_to_prog.get(g.patient_id, 0) for g in graphs])

    if not args.skip_train:
        print('\n[Step 4/6] Training...')
        print(f'  Pre-caching graphs on {device}...')
        for i in range(len(graphs)):
            graphs[i] = graphs[i].to(device)

        from train import Trainer
        trainer = Trainer(CONFIG, device=device)
        results = trainer.run_cv(
            graphs, labels, adrc_ids, prog_labels, in_dims,
            n_folds=CONFIG.get('n_folds', 5),
            n_epochs=args.n_epochs,
            patience=args.patience,
            out_dir=out_dir)

        import pandas as pd
        df_results = pd.DataFrame(results['fold_results'])
        print('\n  Aggregated Results:')
        for metric in ['auroc', 'f1_macro', 'bal_acc']:
            if metric in df_results.columns:
                m = df_results[metric]
                print(f'    {metric:15s}: {m.mean():.4f} +/- {m.std():.4f}')

        print('\n[Step 5/6] Baselines...')
        from explain import Explainer
        temp_explainer = Explainer(None, graphs, feat_defs, device)
        X_flat, y_flat = temp_explainer.extract_flat_features()

        from evaluate import Evaluator
        evaluator = Evaluator(CONFIG)
        bl_results, bl_preds = evaluator.run_baselines(
            X_flat, y_flat, groups=adrc_ids)
        evaluator.save_results(bl_results, bl_preds, out_dir)
        for name, metrics in bl_results.items():
            print(f'    {name:25s}: AUROC={metrics["auroc"]:.4f}')

    if not args.skip_explain:
        print('\n[Step 6/6] Explainability...')
        from model import CogGODE
        import glob, torch

        with open(os.path.join(out_dir, 'model_in_dims.pkl'), 'rb') as f:
            in_dims = pickle.load(f)

        model = CogGODE(
            in_dims=in_dims, hidden=CONFIG['hidden_dim'],
            n_classes=CONFIG['n_classes'], n_concepts=CONFIG['n_concepts'],
            ode_method=CONFIG['ode_method'], dropout=0.0
        ).to(device)

        ckpt_dir = os.path.join(out_dir, 'checkpoints')
        ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, 'best_fold*.pt')))
        if ckpt_files:
            model.load_state_dict(
                torch.load(ckpt_files[0], weights_only=True), strict=False)
        model.eval()

        from explain import Explainer
        explainer = Explainer(model, graphs, feat_defs, device)
        explainer.compute_concept_predictions()
        concept_metrics = explainer.compute_concept_metrics()
        for name, m in concept_metrics.items():
            print(f'    {name:25s}: MAE={m["mae"]:.3f}, r={m["correlation"]:.3f}')
        explainer.save_artifacts(out_dir)

    print('\n' + '=' * 60)
    print('PIPELINE COMPLETE')
    print('=' * 60)
    print(f'Results saved to: {out_dir}/')


if __name__ == '__main__':
    main()
