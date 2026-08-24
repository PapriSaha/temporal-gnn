import os
import pickle

import torch
import numpy as np
from torch_geometric.data import HeteroData
from sklearn.neighbors import NearestNeighbors


class GraphBuilder:

    def __init__(self, cog_feats, func_feats, risk_feats, concept_names):
        self.cog_feats = cog_feats
        self.func_feats = func_feats
        self.risk_feats = risk_feats
        self.concept_names = concept_names

    def build_patient_graph(self, patient_df, patient_id, label):
        cog_present = [c for c in self.cog_feats if c in patient_df.columns]
        func_present = [c for c in self.func_feats if c in patient_df.columns]
        risk_present = [c for c in self.risk_feats if c in patient_df.columns]

        cog_x = torch.tensor(patient_df[cog_present].values, dtype=torch.float32)
        func_x = torch.tensor(patient_df[func_present].values, dtype=torch.float32)
        risk_x = torch.tensor(patient_df[risk_present].values, dtype=torch.float32)

        T = len(patient_df)
        temporal_edges = self._build_temporal_edges(T)
        cross_edges = self._build_cross_domain_edges(T)

        data = HeteroData()
        data['cognitive'].x = cog_x
        data['functional'].x = func_x
        data['risk'].x = risk_x

        data['cognitive', 'temporal', 'cognitive'].edge_index = temporal_edges
        data['functional', 'temporal', 'functional'].edge_index = temporal_edges

        for src, dst in [('cognitive', 'functional'), ('cognitive', 'risk'), ('functional', 'risk')]:
            data[src, 'cross_domain', dst].edge_index = cross_edges

        if 'VISITYR' in patient_df.columns and 'VISITMO' in patient_df.columns:
            timestamps = patient_df['VISITYR'].values + (patient_df['VISITMO'].values - 1) / 12.0
            timestamps = timestamps - timestamps[0]
        else:
            timestamps = np.arange(T, dtype=np.float32)
        data.timestamps = torch.tensor(timestamps, dtype=torch.float32)

        data.y = torch.tensor(label, dtype=torch.long)
        data.patient_id = patient_id
        data.n_visits = T
        if 'NACCADC' in patient_df.columns:
            data.adrc = int(patient_df['NACCADC'].iloc[0])
        else:
            data.adrc = 0

        concept_labels = self._compute_concept_labels(patient_df)
        data.concept_labels = concept_labels

        return data

    def _build_temporal_edges(self, T):
        if T < 2:
            return torch.zeros((2, 0), dtype=torch.long)
        src = list(range(T - 1)) + list(range(1, T))
        dst = list(range(1, T)) + list(range(T - 1))
        return torch.tensor([src, dst], dtype=torch.long)

    def _build_cross_domain_edges(self, T):
        src = list(range(T))
        dst = list(range(T))
        return torch.tensor([src, dst], dtype=torch.long)

    def _compute_concept_labels(self, patient_df):
        last = patient_df.iloc[-1]
        concepts = []

        concepts.append(last.get('GLOBAL_COG_RAW_SCORE', 0.0))
        concepts.append(last.get('MEMORY_RAW_SCORE', 0.0))

        exec_feats = ['TRAILA', 'TRAILB']
        exec_vals = [last.get(f, 0.0) for f in exec_feats]
        concepts.append(np.nanmean(exec_vals) if any(not np.isnan(v) for v in exec_vals if isinstance(v, float)) else 0.0)

        concepts.append(last.get('NAMING_RAW_SCORE', 0.0))
        concepts.append(last.get('FAQ_TOTAL', 0.0))

        npiq = ['ANXSEV', 'APASEV', 'AGITSEV', 'DELSEV', 'HALLSEV', 'IRRSEV', 'MOTSEV']
        npiq_vals = [last.get(f, 0.0) for f in npiq]
        concepts.append(np.nanmean([v for v in npiq_vals if not (isinstance(v, float) and np.isnan(v))]) if npiq_vals else 0.0)

        vasc = ['DIABETES', 'HYPERTEN', 'HYPERCHO', 'CVAFIB', 'CBSTROKE']
        vasc_vals = [last.get(f, 0.0) for f in vasc]
        concepts.append(np.nanmean([v for v in vasc_vals if not (isinstance(v, float) and np.isnan(v))]) if vasc_vals else 0.0)

        concepts.append(last.get('NACCNE4S', 0.0) / 2.0)

        concepts = [0.0 if (isinstance(v, float) and np.isnan(v)) else float(v) for v in concepts]
        # Normalize z-scores to [0, 1] for BCELoss compatibility
        concepts_tensor = torch.tensor(concepts, dtype=torch.float32)
        return torch.sigmoid(concepts_tensor)

    def build_all_graphs(self, df):
        graphs = []
        errors = []
        for pid, group in df.groupby('NACCID'):
            label = group['LABEL'].iloc[-1]
            try:
                g = self.build_patient_graph(group, pid, label)
                graphs.append(g)
            except Exception as e:
                errors.append((pid, str(e)))
        return graphs, errors

    def build_similarity_edges(self, graphs, k=15):
        feats = []
        for g in graphs:
            f = torch.cat([g['cognitive'].x[-1], g['functional'].x[-1], g['risk'].x[-1]]).numpy()
            feats.append(f)
        feats = np.stack(feats)

        k = min(k, len(feats) - 1)
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        nn_model.fit(feats)
        distances, indices = nn_model.kneighbors(feats)

        src, dst = [], []
        for i in range(len(graphs)):
            for j in indices[i][1:]:
                src.extend([i, j])
                dst.extend([j, i])
        return torch.tensor([src, dst], dtype=torch.long)

    def save_graphs(self, graphs, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(graphs, os.path.join(out_dir, 'patient_graphs.pt'))
        concept_labels = np.stack([g.concept_labels.numpy() for g in graphs])
        np.save(os.path.join(out_dir, 'concept_labels.npy'), concept_labels)
        in_dims = {
            'cognitive': graphs[0]['cognitive'].x.shape[1],
            'functional': graphs[0]['functional'].x.shape[1],
            'risk': graphs[0]['risk'].x.shape[1],
        }
        with open(os.path.join(out_dir, 'model_in_dims.pkl'), 'wb') as f:
            pickle.dump(in_dims, f)
        return in_dims
