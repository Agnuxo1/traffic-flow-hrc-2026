import numpy as np
PANELS2=['D12_I5_N', 'D12_I5_S', 'D7_I10_E', 'D7_I10_W', 'D7_I210_E', 'D7_I210_W', 'D7_I405_N', 'D7_I405_S']
FAMILY={'D12_I5_N': 'D12_I5', 'D12_I5_S': 'D12_I5', 'D7_I10_E': 'D7_I10', 'D7_I10_W': 'D7_I10', 'D7_I210_E': 'D7_I210', 'D7_I210_W': 'D7_I210', 'D7_I405_N': 'D7_I405', 'D7_I405_S': 'D7_I405'}
H=6

def window_ious(pred, y, mask, group):
    """pred bool [N,H]; official window IoU over ELIGIBLE cells; empty union -> 1.0; no eligible cell -> 0.0."""
    el = mask > 0; tr = y > .5
    inter = ((pred & tr) & el).sum(1); union = ((pred | tr) & el).sum(1)
    G = int(group.max()) + 1
    I = np.bincount(group, weights=inter, minlength=G); U = np.bincount(group, weights=union, minlength=G)
    C = np.bincount(group, weights=el.sum(1), minlength=G)
    return np.where(C == 0, 0.0, np.where(U > 0, I / np.maximum(U, 1), 1.0))

def exact_score(pred, y, mask, group, panel, cond):
    """window -> panel x condition -> panel -> family -> overall, equal means (published aggregation)."""
    iou = window_ious(pred, y, mask, group)
    ug, first = np.unique(group, return_index=True)
    w_panel = panel[first]; w_cond = cond[first]; w_iou = iou[ug]
    per_panel = {}; per_cond = {0: [], 1: []}
    for p in np.unique(w_panel):
        parts = []
        for c in (0, 1):
            sel = (w_panel == p) & (w_cond == c)
            if sel.any():
                parts.append(w_iou[sel].mean()); per_cond[c].append(parts[-1])
        per_panel[PANELS2[p] if p < len(PANELS2) else str(p)] = float(np.mean(parts))
    fam = {}
    for name, v in per_panel.items():
        fam.setdefault(FAMILY.get(name, name), []).append(v)
    fam_mean = {k: float(np.mean(v)) for k, v in fam.items()}
    return dict(overall=float(np.mean(list(fam_mean.values()))), per_family=fam_mean, per_panel=per_panel,
                per_condition={'queue_onset': float(np.mean(per_cond[0])) if per_cond[0] else None,
                               'queue_ongoing': float(np.mean(per_cond[1])) if per_cond[1] else None},
                n_windows=int(len(ug)))

def model_for(torch, features, cfg):
    nn = torch.nn
    layers = []; dim = features
    for _ in range(cfg['depth']):
        layers.extend([nn.Linear(dim, cfg['width']), nn.LayerNorm(cfg['width']), nn.SiLU(), nn.Dropout(cfg['dropout'])]); dim = cfg['width']
    layers.append(nn.Linear(dim, H))
    return nn.Sequential(*layers)

def normalised(x, mean, std):
    return np.clip((x - mean) / std, -10, 10).astype('float32')
