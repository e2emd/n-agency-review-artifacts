"""Measure saved-output Python postprocessing, separately from GPU generation."""
import argparse
import gc
import json
import platform
import random
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from audit_runtime_v3 import parse_proposal, v2_public_surface, v3_public_surface_controller
from naacl_deep_analysis import MODELS, ROOT, read, sha256, write
from naacl_family_ablation import VARIANTS, surface_for
from reanalyze_runtime_v2 import control_runtime_v2


def postprocess(row, controller):
    """Use the existing parsing, policy, and public-field builders; do not score."""
    proposal = parse_proposal(row)
    if controller == 'native':
        return v2_public_surface(row, proposal)
    if controller == 'v2':
        _, _, control, action, line = control_runtime_v2(row)
        controlled = {**row, 'control_decision_v2': control,
                      'public_action_v2': action, 'public_output_v2': line}
        return v2_public_surface(controlled, proposal)
    if controller == 'v3':
        return v3_public_surface_controller(row, proposal)
    raise ValueError(controller)


def policy_orders(repeats, seed):
    """Each five-pass block balances every condition across execution positions."""
    if repeats < 5 or repeats % len(VARIANTS):
        raise ValueError('repeats must be a positive multiple of five')
    rng = random.Random(seed)
    orders = []
    for _ in range(repeats // len(VARIANTS)):
        base = list(range(len(VARIANTS)))
        rng.shuffle(base)
        orders.extend(base[i:] + base[:i] for i in range(len(base)))
    return orders


def run_batch(rows, controller):
    for row in rows:
        postprocess(row, controller)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    index = int(position)
    return ordered[index] + (ordered[min(index + 1, len(ordered) - 1)] - ordered[index]) * (position - index)


def summarize(timings, repeats):
    groups = defaultdict(lambda: defaultdict(list))
    for row in timings:
        for model in [row['model'], 'ALL']:
            groups[(model, row['variant'])][row['repeat']].append(row)
    output = []
    for model in [m for _, m in MODELS] + ['ALL']:
        for variant, _, _ in VARIANTS:
            rounds = groups[(model, variant)]
            assert len(rounds) == repeats
            n = {sum(r['n'] for r in rows) for rows in rounds.values()}
            assert len(n) == 1
            count = n.pop()
            means = [sum(r['elapsed_ns'] for r in rows) / count / 1000
                     for rows in rounds.values()]
            output.append(dict(model=model, variant=variant, n_unique=count,
                               measured_passes=repeats, timed_calls=count * repeats,
                               mean_us=statistics.mean(means),
                               sd_round_mean_us=statistics.stdev(means),
                               median_round_mean_us=statistics.median(means),
                               p05_round_mean_us=percentile(means, .05),
                               p95_round_mean_us=percentile(means, .95)))
    return output


def run(results, warmups=5, repeats=30, seed=20261101):
    orders = policy_orders(repeats, seed)
    if warmups < 1:
        raise ValueError('At least one warmup is required')
    paths = [results / f'pag_eval_llamacpp_{slug}{suffix}.csv'
             for slug, _ in MODELS for suffix in ['_full', '_schema_policy_full']]
    hashes = {p.name: sha256(p) for p in paths}
    data = {}
    for slug, model in MODELS:
        rows = read(results / f'pag_eval_llamacpp_{slug}_full.csv')
        rows += read(results / f'pag_eval_llamacpp_{slug}_schema_policy_full.csv')
        for condition in ['safety_prompt', 'schema_only', 'schema_policy']:
            selected = sorted((r for r in rows if r['condition'] == condition),
                              key=lambda r: (r['prompt_id'], r['run']))
            assert len(selected) == 1800
            data[(model, condition)] = selected
    checked = 0
    for _, condition, controller in VARIANTS:
        for _, model in MODELS:
            for row in data[(model, condition)]:
                unchanged = dict(row)
                assert postprocess(row, controller) == surface_for(row, controller)
                assert unchanged == row
                checked += 1
    print(f'Equivalent public fields verified for {checked:,} views.', flush=True)
    for _ in range(warmups):
        for _, condition, controller in VARIANTS:
            for _, model in MODELS:
                run_batch(data[(model, condition)], controller)
    timings = []
    started = datetime.now(timezone.utc).isoformat()
    for repeat, order in enumerate(orders, 1):
        rng = random.Random(seed + repeat)
        indices = list(range(1800))
        rng.shuffle(indices)
        model_order = [m for _, m in MODELS]
        rng.shuffle(model_order)
        # All list construction, validation, and disk I/O stay outside the timer.
        shuffled = {key: [rows[i] for i in indices] for key, rows in data.items()}
        for position, index in enumerate(order):
            variant, condition, controller = VARIANTS[index]
            for model in model_order:
                rows = shuffled[(model, condition)]
                start = time.perf_counter_ns()
                run_batch(rows, controller)
                elapsed = time.perf_counter_ns() - start
                timings.append(dict(repeat=repeat, position=position, model=model,
                                    variant=variant, n=len(rows), elapsed_ns=elapsed,
                                    mean_us=elapsed / len(rows) / 1000))
        if repeat % 5 == 0:
            print(f'Measured {repeat}/{repeats} counterbalanced passes.', flush=True)
    summary = summarize(timings, repeats)
    assert hashes == {p.name: sha256(p) for p in paths}
    write(results / 'pag_naacl_postprocessing_passes.csv', timings)
    write(results / 'pag_naacl_postprocessing_summary.csv', summary)
    try:
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor() or platform.machine()
    code_files = ['naacl_postprocessing_benchmark.py', 'naacl_family_ablation.py',
                  'reanalyze_runtime_v2.py', 'audit_runtime_v3.py', 'eval_pag.py']
    manifest = dict(
        model_generations=0, started_utc=started, input_sha256=hashes,
        code_sha256={name: sha256(Path(__file__).parent / name) for name in code_files},
        equivalence_checked_views=checked, unique_outputs=27000,
        observations_per_variant=9000, warmup_passes=warmups, measured_passes=repeats,
        seed=seed, cpu=cpu, os=f'{platform.system()} {platform.release()}',
        python=platform.python_version(), process_threads=1, affinity_pinned=False,
        gc_enabled=gc.isenabled(), timer='perf_counter_ns (wall clock)',
        timer_resolution_s=time.get_clock_info('perf_counter').resolution,
        timed_scope='Python saved-output parsing, policy enforcement, public-field construction; V2 uses its existing separate policy parser.',
        excluded='Model generation, network, disk I/O, evaluation-only scoring, GUI painting, and timed setup.',
        uncertainty='SD and p05/p95 of 30 pass means, not per-request tail percentiles or confidence intervals.',
        comparability='CPU replay timings are separate from the original GPU generation times; do not sum across devices or claim end-to-end deployment speed.',
        policy='V3 retains its benchmark-conditioned guard; no changes to policy decisions or original outputs.')
    (results / 'pag_naacl_postprocessing_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps([r for r in summary if r['model'] == 'ALL'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--seed', type=int, default=20261101)
    args = parser.parse_args()
    run(args.results_dir, args.warmups, args.repeats, args.seed)
