"""Train actual TCN using only full real/full synthetic public video windows.

No test set enters optimization, fusion or calibration. Whole-video labels are
never assigned to windows of a locally edited video. Checkpoints record input
hashes and held-out calibration. Cold feature extraction is not live detection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.contracts import ROOT, load_config, atomic_json, sha256
from backend.models import Models, TemporalNet
from backend.media import sample_video


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--extract-only', action='store_true')
    p.add_argument('--epochs', type=int, default=40)
    args = p.parse_args()
    seed = 102411
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    config = load_config()
    data = ROOT / 'data/public_genvideo_v1'
    manifest_path = data / 'manifest.json'
    rows = json.loads(manifest_path.read_text(encoding='utf-8'))
    frozen_manifest = [r for r in rows if r['split'] in ['train', 'calibration']]
    manifest_hash = hashlib.sha256(json.dumps(frozen_manifest, sort_keys=True).encode()).hexdigest()
    cache = ROOT / 'data/temporal_features_pts_v2'
    cache.mkdir(parents=True, exist_ok=True)
    models = Models(config)
    models.load(generation_only=True)
    if models.errors:
        raise RuntimeError(models.errors)
    examples, failures = [], []
    for idx, row in enumerate(rows):
        # Frozen test frames are never even decoded by the training process.
        if row['split'] not in ['train', 'calibration']:
            continue
        feature_file = cache / (row['sample_id'] + '.npz')
        try:
            if not feature_file.exists():
                frames, windows, meta = sample_video(data / row['relative_path'], config)
                features, risks, _ = models.encode([f.image for f in frames])
                window_features = np.asarray([features[w['frame_ids']] for w in windows], np.float32)
                masks = np.asarray([w['valid_mask'] for w in windows], np.float32)
                np.savez_compressed(feature_file, features=window_features, masks=masks,
                                    source_sha256=row['sha256'], model='OpenAICLIPViTL14QuickGELU_fp16',
                                    split=row['split'], label=int(row['truth_label']=='fake'))
            with np.load(feature_file) as stored:
                if str(stored['source_sha256']) != row['sha256']:
                    raise ValueError('Source hash changed since feature cache creation')
                for f, mask in zip(stored['features'], stored['masks']):
                    if mask.sum() >= 4:
                        examples.append((f, mask, int(row['truth_label']=='fake'), row['split'], row['sample_id']))
            print('features', idx + 1, row['sample_id'], row['split'], flush=True)
        except Exception as exc:
            failures.append({'sample_id': row['sample_id'], 'error': str(exc), 'split': row['split']})
            print('FAIL', row['sample_id'], repr(exc), flush=True)
    atomic_json(ROOT / 'runs/temporal_feature_failures.json', failures)
    if args.extract_only:
        return
    del models
    torch.cuda.empty_cache()
    train = [e for e in examples if e[3] == 'train']
    calibration = [e for e in examples if e[3] == 'calibration']
    if min(len(train), len(calibration)) < 10 or len({e[2] for e in train}) != 2:
        raise RuntimeError('Not enough traceable training and independent calibration examples')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = TemporalNet().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=.01)
    y_train = torch.tensor([e[2] for e in train], dtype=torch.float32, device=device)
    pos_weight = (1-y_train).sum() / y_train.sum().clamp_min(1)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_x = torch.from_numpy(np.asarray([e[0] for e in train])).to(device)
    train_m = torch.from_numpy(np.asarray([e[1] for e in train])).to(device)
    cal_x = torch.from_numpy(np.asarray([e[0] for e in calibration])).to(device)
    cal_m = torch.from_numpy(np.asarray([e[1] for e in calibration])).to(device)
    cal_y = np.asarray([e[2] for e in calibration])
    history = []
    for epoch in range(args.epochs):
        net.train()
        order = torch.randperm(len(train), device=device)
        losses = []
        for start in range(0, len(order), 32):
            indices = order[start:start+32]
            logits = net(train_x[indices], train_m[indices])
            loss = criterion(logits, y_train[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({'epoch': epoch + 1, 'training_loss': float(np.mean(losses))})
        print(history[-1], flush=True)
    net.eval()
    with torch.inference_mode():
        cal_pred = net(cal_x, cal_m).sigmoid().cpu().numpy()
        # Real order ablation is assessed on the same frames. This does not
        # transform unlabelled windows into temporal ground truth.
        permutation = torch.tensor([0,4,1,6,2,7,3,5],device=device)
        shuffled = net(cal_x[:, permutation], cal_m[:, permutation]).sigmoid().cpu().numpy()
    metrics = {'calibration_auc': float(roc_auc_score(cal_y, cal_pred)),
               'calibration_ap': float(average_precision_score(cal_y, cal_pred)),
               'calibration_accuracy_at_0_5': float(np.mean((cal_pred >= .5) == cal_y)),
               'shuffled_order_mean_abs_score_change': float(np.mean(np.abs(cal_pred-shuffled))),
               'shuffled_order_auc': float(roc_auc_score(cal_y, shuffled)),
               'calibration_windows': len(calibration), 'training_windows': len(train),
               'failed_public_inputs': len(failures), 'test_used': False}
    checkpoint = {'state_dict': {k:v.cpu() for k,v in net.state_dict().items()}, 'trained': True,
                  'model': 'CLIP temporal residual TCN', 'architecture':'centered_temporal_tcn_v1', 'training_manifest_sha256': manifest_hash,
                  'training_source': 'GenVideo-Val deterministic development partition; not contest samples',
                  'training_sample_ids': sorted({e[4] for e in train}),
                  'calibration_sample_ids': sorted({e[4] for e in calibration}),
                  'metrics': metrics, 'seed': seed, 'epochs': args.epochs,
                  'labels': 'Full real/full generated video labels only; no local-edit window pseudo-labels',
                  'score_kind': 'uncalibrated_risk'}
    torch.save(checkpoint, ROOT / 'models/temporal_tcn.pth')
    atomic_json(ROOT / 'runs/temporal_training.json', {'metrics': metrics, 'history': history,
                'checkpoint_sha256': sha256(ROOT / 'models/temporal_tcn.pth'), 'training_manifest_sha256': manifest_hash})
    print('Completed actual TCN training:', metrics, flush=True)


if __name__ == '__main__':
    main()
