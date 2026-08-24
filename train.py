import os
import time
import pickle

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

from model import CogGODE, CogGODELoss, FocalLoss, compute_weights


class Trainer:

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.accum_steps = 32
        self.epoch_train_samples = 6000
        self.epoch_val_samples = 2000

    def stratified_subsample(self, indices, all_labels, n_samples, rng):
        sub_labels = all_labels[indices]
        classes = np.unique(sub_labels)
        selected = []
        for cls in classes:
            cls_idx = indices[sub_labels == cls]
            n_cls = max(1, int(n_samples * len(cls_idx) / len(indices)))
            n_cls = min(n_cls, len(cls_idx))
            selected.extend(rng.choice(cls_idx, n_cls, replace=False).tolist())
        rng.shuffle(selected)
        return selected

    def train_one_epoch(self, model, train_indices, graphs, labels, prog_labels,
                        criterion, optimizer, rng=None):
        model.train()
        total_loss, n, nfe_list, correct = 0, 0, [], 0
        optimizer.zero_grad()

        if rng is None:
            rng = np.random.RandomState()
        if len(train_indices) > self.epoch_train_samples:
            epoch_idx = self.stratified_subsample(
                train_indices, labels, self.epoch_train_samples, rng)
        else:
            epoch_idx = list(train_indices)
            rng.shuffle(epoch_idx)

        for step, gi in enumerate(epoch_idx):
            g = graphs[gi]
            try:
                out = model(g, ode_method_override='euler')
            except RuntimeError as e:
                if 'underflow' in str(e) or 'diverge' in str(e):
                    continue
                raise

            tgt = {
                'label': g.y,
                'risk_label': torch.tensor(
                    g.y.item() > 0, dtype=torch.float32, device=self.device),
                'concept_labels': g.concept_labels,
            }
            if prog_labels is not None:
                tgt['prog_label'] = torch.tensor(
                    float(prog_labels[gi]), dtype=torch.float32, device=self.device)

            loss, _ = criterion(out, tgt)
            loss = loss / self.accum_steps
            loss.backward()

            total_loss += loss.item() * self.accum_steps
            correct += (out['logits'].argmax().item() == g.y.item())
            nfe_list.append(out.get('nfe', 0))
            n += 1

            if (step + 1) % self.accum_steps == 0 or (step + 1) == len(epoch_idx):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        return total_loss / max(n, 1), correct / max(n, 1), np.mean(nfe_list)

    @torch.no_grad()
    def evaluate(self, model, eval_indices, graphs, labels, prog_labels,
                 ode_method='euler', max_samples=None, rng=None):
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        nfe_list = []

        if max_samples and len(eval_indices) > max_samples:
            if rng is None:
                rng = np.random.RandomState(0)
            eval_idx = self.stratified_subsample(
                np.array(eval_indices), labels, max_samples, rng)
        else:
            eval_idx = eval_indices

        for gi in eval_idx:
            g = graphs[gi]
            try:
                out = model(g, ode_method_override=ode_method)
            except RuntimeError:
                continue
            y_true.append(g.y.item())
            y_pred.append(out['logits'].argmax().item())
            y_prob.append(F.softmax(out['logits'], dim=-1).cpu().numpy())
            nfe_list.append(out.get('nfe', 0))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        metrics = {
            'auroc': roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'bal_acc': balanced_accuracy_score(y_true, y_pred),
            'mean_nfe': np.mean(nfe_list),
        }
        return metrics, y_true, y_pred, y_prob

    def create_model(self, in_dims):
        return CogGODE(
            in_dims=in_dims,
            hidden=self.config['hidden_dim'],
            n_classes=self.config['n_classes'],
            n_concepts=self.config['n_concepts'],
            ode_method='euler',
            dropout=self.config['dropout'],
        ).to(self.device)

    def run_cv(self, graphs, labels, adrc_ids, prog_labels, in_dims,
               n_folds=5, n_epochs=30, patience=7, out_dir='./results'):
        outer_cv = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=42)
        outer_folds = list(outer_cv.split(
            np.arange(len(graphs)), labels, groups=adrc_ids))

        all_fold_results = []
        all_fold_predictions = []
        computational_costs = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(outer_folds):
            fold_start = time.time()

            inner_labels = labels[train_val_idx]
            inner_cv = StratifiedKFold(
                n_splits=4, shuffle=True, random_state=fold_idx)
            inner_train_rel, inner_val_rel = next(
                inner_cv.split(train_val_idx, inner_labels))
            actual_train_idx = train_val_idx[inner_train_rel]
            actual_val_idx = train_val_idx[inner_val_rel]

            class_weights = compute_weights(
                labels[actual_train_idx]).to(self.device)

            model = self.create_model(in_dims)
            criterion = CogGODELoss(
                class_weights=class_weights, use_focal=True,
                lam=(1.0, 0.5, 0.01, 0.3, 0.3))

            base_lr = self.config.get('lr', 1e-3)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=base_lr * 0.1,
                weight_decay=self.config.get('weight_decay', 1e-4))
            warmup_epochs = 3
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0,
                total_iters=warmup_epochs)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs - warmup_epochs,
                eta_min=base_lr * 0.01)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs])

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            best_val_auroc = 0
            patience_counter = 0
            epoch_times = []
            rng = np.random.RandomState(fold_idx * 1000)

            for epoch in range(n_epochs):
                epoch_start = time.time()
                train_loss, train_acc, train_nfe = self.train_one_epoch(
                    model, actual_train_idx, graphs, labels, prog_labels,
                    criterion, optimizer, rng=rng)

                val_rng = np.random.RandomState(epoch)
                val_metrics, _, _, _ = self.evaluate(
                    model, actual_val_idx, graphs, labels, prog_labels,
                    ode_method='euler', max_samples=self.epoch_val_samples,
                    rng=val_rng)
                scheduler.step()

                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)

                print(f'  Fold {fold_idx} Ep {epoch:3d} | '
                      f'Loss={train_loss:.4f} | '
                      f'Val AUROC={val_metrics["auroc"]:.4f} | '
                      f'{epoch_time:.1f}s')

                if val_metrics['auroc'] > best_val_auroc:
                    best_val_auroc = val_metrics['auroc']
                    ckpt_dir = os.path.join(out_dir, 'checkpoints')
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(model.state_dict(),
                               os.path.join(ckpt_dir, f'best_fold{fold_idx}.pt'))
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            ckpt_path = os.path.join(out_dir, 'checkpoints', f'best_fold{fold_idx}.pt')
            model.load_state_dict(torch.load(ckpt_path, weights_only=True))

            inf_start = time.time()
            test_metrics, y_true_test, y_pred_test, y_prob_test = self.evaluate(
                model, test_idx, graphs, labels, prog_labels,
                ode_method='dopri5')
            inf_time = time.time() - inf_start
            fold_time = time.time() - fold_start

            peak_mem = (torch.cuda.max_memory_allocated() / 1e9
                        if torch.cuda.is_available() else 0)

            test_metrics['fold'] = fold_idx
            all_fold_results.append(test_metrics)
            all_fold_predictions.append({
                'y_true': y_true_test, 'y_pred': y_pred_test,
                'y_prob': y_prob_test, 'test_idx': test_idx,
            })
            computational_costs.append({
                'fold': fold_idx,
                'train_time_s': fold_time - inf_time,
                'inference_time_s': inf_time,
                'inference_per_patient_ms': inf_time / len(test_idx) * 1000,
                'epochs_trained': epoch + 1,
                'mean_epoch_time_s': np.mean(epoch_times),
                'peak_gpu_memory_gb': peak_mem,
                'mean_nfe': test_metrics['mean_nfe'],
            })

        results = self._aggregate_results(
            all_fold_results, all_fold_predictions, computational_costs)
        self._save_results(results, out_dir)
        return results

    def _aggregate_results(self, fold_results, fold_predictions, costs):
        y_true_all = np.concatenate([p['y_true'] for p in fold_predictions])
        y_pred_all = np.concatenate([p['y_pred'] for p in fold_predictions])
        y_prob_all = np.concatenate([p['y_prob'] for p in fold_predictions])
        return {
            'fold_results': fold_results,
            'fold_predictions': fold_predictions,
            'y_true_test': y_true_all,
            'y_pred_test': y_pred_all,
            'y_prob_test': y_prob_all,
            'computational_costs': costs,
        }

    def _save_results(self, results, out_dir):
        os.makedirs(os.path.join(out_dir, 'tables'), exist_ok=True)
        with open(os.path.join(out_dir, 'training_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        pd.DataFrame(results['fold_results']).to_csv(
            os.path.join(out_dir, 'tables', 'cv_fold_results.csv'), index=False)
        pd.DataFrame(results['computational_costs']).to_csv(
            os.path.join(out_dir, 'tables', 'computational_costs.csv'), index=False)
