import os
import pickle

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

try:
    import shap
except ImportError:
    shap = None


class Explainer:

    def __init__(self, model, graphs, feature_defs, device='cuda'):
        self.model = model
        self.graphs = graphs
        self.feature_defs = feature_defs
        self.device = device
        self.concept_names = feature_defs['concept_names']
        self.cog_feats = feature_defs['COG_FEATS']
        self.func_feats = feature_defs['FUNC_FEATS']
        self.risk_feats = feature_defs['RISK_FEATS']
        self.concept_preds = None
        self.concept_trues = None

    def compute_concept_predictions(self, n_eval=2000):
        self.model.eval()
        preds, trues = [], []
        n_eval = min(n_eval, len(self.graphs))

        with torch.no_grad():
            for g in self.graphs[:n_eval]:
                g = g.to(self.device)
                try:
                    out = self.model(g)
                    preds.append(out['concepts'].cpu().numpy())
                    trues.append(g.concept_labels.cpu().numpy())
                except Exception:
                    continue

        self.concept_preds = np.stack(preds)
        self.concept_trues = np.stack(trues)
        return self.concept_preds, self.concept_trues

    def compute_concept_metrics(self):
        if self.concept_preds is None:
            self.compute_concept_predictions()

        metrics = {}
        for i, name in enumerate(self.concept_names):
            mae = mean_absolute_error(
                self.concept_trues[:, i], self.concept_preds[:, i])
            corr = np.corrcoef(
                self.concept_trues[:, i], self.concept_preds[:, i])[0, 1]
            metrics[name] = {'mae': mae, 'correlation': corr}
        return metrics

    def extract_flat_features(self, n_eval=2000):
        n_eval = min(n_eval, len(self.graphs))
        X, Y = [], []
        for g in self.graphs[:n_eval]:
            feat = torch.cat([
                g['cognitive'].x[-1],
                g['functional'].x[-1],
                g['risk'].x[-1]
            ]).cpu().numpy()
            X.append(feat)
            Y.append(g.y.item())
        return np.stack(X), np.array(Y)

    def run_shap_analysis(self, n_eval=2000, n_shap=500):
        if shap is None:
            return None

        X, Y = self.extract_flat_features(n_eval)
        all_feat_names = self.cog_feats + self.func_feats + self.risk_feats
        feat_names = [n[:20] for n in all_feat_names[:X.shape[1]]]

        surrogate = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42)
        surrogate.fit(X, Y)

        explainer_obj = shap.TreeExplainer(surrogate)
        shap_values = explainer_obj.shap_values(X[:n_shap])

        if isinstance(shap_values, list):
            shap_per_class = shap_values
        else:
            shap_per_class = [
                shap_values[:, :, c] for c in range(shap_values.shape[2])]

        shap_abs_mean = np.array(
            [np.abs(sv).mean(axis=0) for sv in shap_per_class])
        shap_overall = shap_abs_mean.mean(axis=0)
        top_k = min(15, X.shape[1])
        top_idx = np.argsort(shap_overall)[::-1][:top_k]

        return {
            'shap_values': shap_per_class,
            'shap_abs_mean': shap_abs_mean,
            'top_feature_indices': top_idx,
            'feature_names': feat_names,
            'X_shap': X[:n_shap],
        }

    def compute_temporal_attention(self, n_eval=500):
        self.model.eval()
        attention_by_class = {0: [], 1: [], 2: []}
        n_eval = min(n_eval, len(self.graphs))

        with torch.no_grad():
            for g in self.graphs[:n_eval]:
                g = g.to(self.device)
                try:
                    out = self.model(g)
                    label = g.y.item()
                    attn = out['attn_weights'].cpu().numpy()
                    attention_by_class[label].append(attn)
                except Exception:
                    continue

        return attention_by_class

    def save_artifacts(self, out_dir):
        os.makedirs(os.path.join(out_dir, 'explanations'), exist_ok=True)
        artifacts = {
            'concept_predictions': self.concept_preds,
            'concept_ground_truth': self.concept_trues,
            'concept_names': self.concept_names,
        }
        path = os.path.join(out_dir, 'explanations', 'explainer_artifacts.pkl')
        with open(path, 'wb') as f:
            pickle.dump(artifacts, f)
