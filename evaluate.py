import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (roc_auc_score, roc_curve, f1_score,
                              balanced_accuracy_score, matthews_corrcoef,
                              confusion_matrix, brier_score_loss,
                              classification_report)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class Evaluator:

    def __init__(self, config):
        self.config = config

    def compute_metrics(self, y_true, y_pred, y_prob):
        metrics = {
            'auroc': roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'bal_acc': balanced_accuracy_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
        }

        n_classes = y_prob.shape[1]
        for c in range(n_classes):
            binary_true = (y_true == c).astype(int)
            if len(np.unique(binary_true)) > 1:
                metrics[f'auroc_class{c}'] = roc_auc_score(binary_true, y_prob[:, c])
                metrics[f'f1_class{c}'] = f1_score(binary_true, (y_pred == c).astype(int))

        return metrics

    def bootstrap_ci(self, y_true, y_prob, y_pred=None, n_bootstrap=1000,
                     ci=0.95, seed=42):
        rng = np.random.RandomState(seed)
        n = len(y_true)
        boot_metrics = []

        for _ in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            bt = y_true[idx]
            bp = y_prob[idx]
            if len(np.unique(bt)) < 2:
                continue
            try:
                auroc = roc_auc_score(bt, bp, multi_class='ovr', average='macro')
                boot_metrics.append(auroc)
            except Exception:
                continue

        boot_metrics = np.array(boot_metrics)
        alpha = (1 - ci) / 2
        return {
            'mean': boot_metrics.mean(),
            'std': boot_metrics.std(),
            'ci_lower': np.percentile(boot_metrics, alpha * 100),
            'ci_upper': np.percentile(boot_metrics, (1 - alpha) * 100),
        }

    def delong_test(self, y_true, prob_a, prob_b):
        try:
            from scipy.stats import norm
            auc_a = roc_auc_score(y_true, prob_a)
            auc_b = roc_auc_score(y_true, prob_b)
            n = len(y_true)
            se = np.sqrt((auc_a * (1 - auc_a) + auc_b * (1 - auc_b)) / n)
            if se < 1e-10:
                return {'auc_a': auc_a, 'auc_b': auc_b, 'z': 0.0, 'p': 1.0}
            z = (auc_a - auc_b) / se
            p = 2 * (1 - norm.cdf(abs(z)))
            return {'auc_a': auc_a, 'auc_b': auc_b, 'z': z, 'p': p}
        except Exception:
            return {'auc_a': 0.0, 'auc_b': 0.0, 'z': 0.0, 'p': 1.0}

    def mcnemar_test(self, correct_a, correct_b):
        b = np.sum((correct_a == 1) & (correct_b == 0))
        c = np.sum((correct_a == 0) & (correct_b == 1))
        if b + c == 0:
            return {'statistic': 0.0, 'p_value': 1.0}
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p = 1 - stats.chi2.cdf(chi2, df=1)
        return {'statistic': chi2, 'p_value': p}

    def run_baselines(self, X, y, groups=None, n_folds=5):
        baselines = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=5, random_state=42),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
        }

        if groups is not None:
            cv = StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(cv.split(X, y, groups=groups))
        else:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=42)
            splits = list(cv.split(X, y))

        results = {}
        predictions = {}

        for name, clf in baselines.items():
            fold_aurocs = []
            all_y_true, all_y_pred, all_y_prob = [], [], []

            for train_idx, test_idx in splits:
                clf_copy = type(clf)(**clf.get_params())
                clf_copy.fit(X[train_idx], y[train_idx])
                y_pred = clf_copy.predict(X[test_idx])
                y_prob = clf_copy.predict_proba(X[test_idx])

                all_y_true.extend(y[test_idx])
                all_y_pred.extend(y_pred)
                all_y_prob.append(y_prob)

                try:
                    auroc = roc_auc_score(
                        y[test_idx], y_prob, multi_class='ovr', average='macro')
                    fold_aurocs.append(auroc)
                except Exception:
                    pass

            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_y_prob = np.vstack(all_y_prob)

            results[name] = self.compute_metrics(
                all_y_true, all_y_pred, all_y_prob)
            results[name]['auroc_mean'] = np.mean(fold_aurocs)
            results[name]['auroc_std'] = np.std(fold_aurocs)
            predictions[name] = {
                'y_true': all_y_true, 'y_pred': all_y_pred, 'y_prob': all_y_prob}

        return results, predictions

    def save_results(self, results, predictions, out_dir):
        os.makedirs(os.path.join(out_dir, 'baselines'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'tables'), exist_ok=True)

        rows = []
        for name, metrics in results.items():
            row = {'model': name}
            row.update(metrics)
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, 'tables', 'baseline_comparison.csv'), index=False)

        with open(os.path.join(out_dir, 'baselines', 'baseline_predictions.pkl'), 'wb') as f:
            pickle.dump(predictions, f)
