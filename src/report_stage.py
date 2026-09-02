import json
import shutil
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import atomic_json


def finalize(p, baseline, kd):
    print('\n' + '=' * 80 + '\nSTAGE 7: FINAL REPORT\n' + '=' * 80)

    bval = baseline['val'].copy()
    kval = kd['val'].copy()
    btrain = baseline['train'].copy()
    ktrain = kd['train'].copy()

    b0 = float(bval.iloc[0]['validation_loss'])
    k0 = float(kval.iloc[0]['validation_loss'])
    if abs(b0 - k0) > 1e-5:
        raise RuntimeError(
            f'baseline/KD initial validation mismatch: {b0} vs {k0}; '
            'the two students did not start from the same effective initialization'
        )

    brow = bval.loc[bval['validation_loss'].idxmin()]
    krow = kval.loc[kval['validation_loss'].idxmin()]
    bbest = float(brow['validation_loss'])
    kbest = float(krow['validation_loss'])
    bstep = int(brow['iteration'])
    kstep = int(krow['iteration'])
    diff = kbest - bbest

    results = pd.DataFrame({
        'metric': ['initial_validation', 'best_validation', 'best_iteration', 'final_validation'],
        'baseline': [b0, bbest, bstep, float(bval.iloc[-1]['validation_loss'])],
        'kd': [k0, kbest, kstep, float(kval.iloc[-1]['validation_loss'])],
    })
    results.to_csv(p['reports'] / 'final_results.csv', index=False)

    btrain['smooth_ce'] = btrain['hard_training_loss'].rolling(5, min_periods=1).mean()
    ktrain['smooth_ce'] = ktrain['hard_training_loss'].rolling(5, min_periods=1).mean()

    fig = plt.figure(figsize=(13, 8))
    plt.plot(btrain['iteration'], btrain['smooth_ce'], alpha=.55, label='Baseline train CE')
    plt.plot(ktrain['iteration'], ktrain['smooth_ce'], alpha=.55, label='KD train CE')
    plt.plot(bval['iteration'], bval['validation_loss'], marker='o', linewidth=2.5, label='Baseline validation')
    plt.plot(kval['iteration'], kval['validation_loss'], marker='o', linewidth=2.5, label='KD validation')
    plt.scatter([bstep], [bbest], s=130)
    plt.scatter([kstep], [kbest], s=130)
    plt.xlabel('Optimizer iteration')
    plt.ylabel('Cross-entropy loss')
    plt.title('CodeParrot 1.5B — Balanced Multi-Teacher Distillation\nAll 8 teachers trained on equal mixtures of all 8 domains')
    plt.grid(alpha=.25)
    plt.legend()
    plt.tight_layout()
    fig.savefig(p['reports'] / 'baseline_vs_kd.svg', format='svg', bbox_inches='tight')
    plt.close(fig)

    diagnostics_path = p['scores'] / 'diagnostics.json'
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else {}
    summary = {
        'complete': True,
        'baseline_initial_validation': b0,
        'kd_initial_validation': k0,
        'baseline_best_validation': bbest,
        'baseline_best_step': bstep,
        'kd_best_validation': kbest,
        'kd_best_step': kstep,
        'kd_minus_baseline_best_validation': diff,
        'winner': 'kd' if diff < 0 else ('baseline' if diff > 0 else 'tie'),
        'teacher_diagnostics': diagnostics,
    }
    if bbest != 0:
        summary['kd_relative_change_percent'] = float(diff / bbest * 100.0)

    atomic_json(p['reports'] / 'summary.json', summary)

    bundle = p['reports'] / 'bundle'
    shutil.rmtree(bundle, ignore_errors=True)
    bundle.mkdir(parents=True, exist_ok=True)
    candidates = [
        p['root'] / 'teacher_domain_balance.csv',
        p['root'] / 'teacher_training.csv',
        p['root'] / 'teacher_domain_nll.csv',
        p['scores'] / 'diagnostics.json',
        p['scores'] / 'oracle_selection.csv',
        p['students'] / 'baseline' / 'training.csv',
        p['students'] / 'baseline' / 'validation.csv',
        p['students'] / 'kd' / 'training.csv',
        p['students'] / 'kd' / 'validation.csv',
        p['reports'] / 'final_results.csv',
        p['reports'] / 'baseline_vs_kd.svg',
        p['reports'] / 'summary.json',
    ]
    for source in candidates:
        if source.exists():
            shutil.copy2(source, bundle / f'{source.parent.name}_{source.name}')

    atomic_json(p['finished'], summary)
    print(json.dumps(summary, indent=2))
    print('[report] COMPLETE')
