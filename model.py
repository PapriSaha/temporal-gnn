from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv
from torchdiffeq import odeint


class HeteroGNNLayer(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.convs = HeteroConv({
            ('cognitive', 'temporal', 'cognitive'):
                GATConv((-1, -1), hidden, heads=4, concat=False, add_self_loops=False),
            ('functional', 'temporal', 'functional'):
                GATConv((-1, -1), hidden, heads=4, concat=False, add_self_loops=False),
            ('cognitive', 'cross_domain', 'functional'):
                SAGEConv((-1, -1), hidden),
            ('cognitive', 'cross_domain', 'risk'):
                SAGEConv((-1, -1), hidden),
            ('functional', 'cross_domain', 'risk'):
                SAGEConv((-1, -1), hidden),
        }, aggr='mean')
        self.norms = nn.ModuleDict({
            t: nn.LayerNorm(hidden) for t in ['cognitive', 'functional', 'risk']
        })

    def forward(self, x_dict, edge_index_dict):
        out = self.convs(x_dict, edge_index_dict)
        return {
            k: F.relu(self.norms[k](v))
            for k, v in out.items() if k in self.norms
        }


class GraphODEFunc(nn.Module):

    def __init__(self, hidden, edge_index_dict=None):
        super().__init__()
        self.edge_index_dict = edge_index_dict
        self.gnn = HeteroGNNLayer(hidden)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.node_shapes = {}
        self.nfe = 0

    def forward(self, t, h_flat):
        self.nfe += 1
        x_dict = self._unflatten(h_flat)
        t_emb = self.time_embed(t.detach().reshape(1, 1))
        x_dict = {k: v + t_emb for k, v in x_dict.items()}
        dx = self.gnn(x_dict, self.edge_index_dict)
        out = {}
        for k in x_dict:
            if k in dx:
                out[k] = dx[k] - x_dict[k]
            else:
                out[k] = torch.zeros_like(x_dict[k])
        return self._flatten(out)

    def reset_nfe(self):
        self.nfe = 0

    def _flatten(self, x_dict):
        return torch.cat([x_dict[k].reshape(-1) for k in sorted(x_dict)], dim=0)

    def _unflatten(self, h):
        out, idx = {}, 0
        for k in sorted(self.node_shapes):
            shape = self.node_shapes[k]
            n = shape[0] * shape[1]
            out[k] = h[idx:idx + n].reshape(shape)
            idx += n
        return out


class CogGODE(nn.Module):

    def __init__(self, in_dims, hidden=128, n_classes=3, n_concepts=8,
                 ode_method='dopri5', dropout=0.3):
        super().__init__()
        self.hidden = hidden
        self.ode_method = ode_method

        self.proj = nn.ModuleDict({
            k: nn.Linear(v, hidden) for k, v in in_dims.items()
        })

        self.enc1 = HeteroGNNLayer(hidden)
        self.enc2 = HeteroGNNLayer(hidden)

        self.ode_func = GraphODEFunc(hidden)

        self.temp_attn = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, n_classes)
        )

        self.progression_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1), nn.Sigmoid()
        )

        self.risk_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid()
        )

        self.concept_head = nn.Linear(hidden, n_concepts)

    def forward(self, data, ode_method_override=None):
        method = ode_method_override or self.ode_method
        x_dict = {}
        for k in ['cognitive', 'functional', 'risk']:
            if k in data.node_types:
                x_dict[k] = self.proj[k](data[k].x)

        ei = {}
        for et in data.edge_types:
            if hasattr(data[et], 'edge_index'):
                ei[et] = data[et].edge_index

        x_dict = self.enc1(x_dict, ei)
        x_dict = self.enc2(x_dict, ei)

        self.ode_func.edge_index_dict = ei
        self.ode_func.node_shapes = {k: tuple(v.shape) for k, v in x_dict.items()}
        self.ode_func.reset_nfe()

        h0 = self.ode_func._flatten(x_dict)
        timestamps = data.timestamps.unique(sorted=True)
        if timestamps.numel() < 2:
            timestamps = torch.tensor([0.0, 1.0], device=h0.device)

        h_traj = odeint(
            self.ode_func, h0, timestamps,
            method=method, rtol=1e-3, atol=1e-4
        )

        T = len(timestamps)
        h_per_t = []
        for i in range(T):
            step_dict = self.ode_func._unflatten(h_traj[i])
            pooled = torch.stack([v.mean(0) for v in step_dict.values()]).mean(0)
            h_per_t.append(pooled)

        h_stack = torch.stack(h_per_t)

        attn = torch.softmax(self.temp_attn(h_stack).squeeze(-1), dim=0)
        h_patient = (attn.unsqueeze(-1) * h_stack).sum(0)

        return {
            'logits': self.classifier(h_patient),
            'progression': self.progression_head(h_patient),
            'risk': self.risk_head(h_patient),
            'concepts': torch.sigmoid(self.concept_head(h_patient)),
            'h_patient': h_patient,
            'attn_weights': attn,
            'h_trajectory': h_traj,
            'nfe': self.ode_func.nfe,
        }


class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce
        return focal.mean()


class CogGODELoss(nn.Module):

    def __init__(self, class_weights=None, use_focal=False,
                 lam=(1.0, 0.5, 0.01, 0.3, 0.3)):
        super().__init__()
        if use_focal:
            self.ce = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.bce = nn.BCELoss()
        self.concept_loss = nn.BCELoss()
        self.prog_loss = nn.BCELoss()
        self.lam = lam

    def forward(self, out, tgt):
        l1 = self.ce(out['logits'].unsqueeze(0), tgt['label'].unsqueeze(0))
        l2 = self.bce(out['risk'].squeeze(), tgt['risk_label'].float())

        h_traj = out['h_trajectory']
        l3 = h_traj.diff(dim=0).pow(2).mean() if h_traj.size(0) > 1 else torch.tensor(0.0, device=out['logits'].device)

        l4 = self.concept_loss(out['concepts'], tgt['concept_labels'])

        l5 = self.prog_loss(out['progression'].squeeze(), tgt['prog_label'].float()) if 'prog_label' in tgt else torch.tensor(0.0, device=out['logits'].device)

        total = (self.lam[0] * l1 + self.lam[1] * l2 +
                 self.lam[2] * l3 + self.lam[3] * l4 + self.lam[4] * l5)

        return total, {
            'ce': l1.item(), 'risk': l2.item(),
            'ode': l3.item() if isinstance(l3, torch.Tensor) else l3,
            'concept': l4.item(), 'progression': l5.item() if isinstance(l5, torch.Tensor) else l5,
        }


def compute_weights(graph_labels):
    c = np.bincount(graph_labels.astype(int), minlength=3)
    w = 1.0 / (c + 1e-8)
    w = w / w.sum() * 3
    return torch.tensor(w, dtype=torch.float32)


def count_parameters(model):
    module_params = defaultdict(int)
    total = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            module_params[name.split('.')[0]] += n
    return total
