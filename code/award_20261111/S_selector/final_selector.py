"""Build reviewable candidates from frozen models. No threshold search, no sealed days, no submission."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from final_logic import archive, build_predictions, combine, describe, digest, load_arm, write_json
from tfb_common import REL

HERE = Path(__file__).resolve().parent


def check_sources():
    manifest = json.loads((HERE / 'source-manifest.json').read_text())
    for name, sha in manifest.items():
        assert digest(HERE / name) == sha, 'Source changed: ' + name
    return digest(HERE / 'source-manifest.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    manifest_sha = check_sources()
    config = json.loads((HERE / 'config/final.json').read_text())
    assert not config['confirmation_opened'] and not config['portal_submission']
    assert datetime.now(timezone.utc) < datetime.fromisoformat(config['hard_stop_utc'].replace('Z', '+00:00')), 'Past the closing limit'
    arms = {name: (load_arm(arm), arm['threshold']) for name, arm in config['arms'].items()}
    frozen = dict(time=datetime.now(timezone.utc).isoformat(), source_manifest_sha256=manifest_sha, config_sha256=digest(HERE / 'config/final.json'),
                  thresholds={k: v['threshold'] for k, v in config['arms'].items()}, confirmation_opened=False)
    write_json(out / 'frozen_choice.json', frozen)
    predictions, summary = build_predictions(arms)
    summary.to_csv(out / 'window_summary.csv', index=False)
    artifacts = {}
    for name, series in predictions.items():
        dest = out / f'traffic_candidate_{name}.csv'
        info = combine(series, dest, config['base']['path'], config['base']['sha256'], REL / 'submission_key.csv')
        packed = archive(dest)
        artifacts[name] = dict(info, csv=str(dest), csv_sha256=digest(dest), zip=str(packed), zip_sha256=digest(packed))
    result = dict(status='PREPARED_FOR_OWNER_REVIEW', frozen=frozen, artifacts=artifacts, predictions=describe(summary),
                  official_improvement_verified=False, submission_sent=False, promoted=False, confirmation_opened=False,
                  code_pdf_deadline_utc='2026-11-11T11:59:00Z')
    write_json(out / 'result.json', result)
    write_json('result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
