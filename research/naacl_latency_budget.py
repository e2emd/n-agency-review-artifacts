"""Audit-qualified generation-budget yield from saved rows; no inference."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json

from naacl_deep_analysis import (MODELS, ROOT, read, write, request_pass, pct,
                                 quantile, bootstrap_weights, sha256)

BUDGETS = [1.0, 1.5, 2.0, 3.0]
POLICIES = ['v2_public_surface', 'v3_public_surface_controller']


def qualified_yield(rows, budget):
    return sum(request_pass(r) and float(r['latency_s']) <= budget for r in rows)


def run(results, bootstraps=2000):
    paths = [results / f'pag_runtime_v3_live_{slug}_full_audit.csv' for slug, _ in MODELS]
    hashes = {p.name: sha256(p) for p in paths}
    groups, categories, curves, endpoints = {}, {}, [], []
    pooled_events = {variant:Counter() for variant in POLICIES}
    pooled_groups = {variant:defaultdict(lambda:[0]*len(BUDGETS)) for variant in POLICIES}
    for (_, model), path in zip(MODELS, paths):
        audit = read(path)
        paired_times = {}
        for variant in POLICIES:
            rows = [r for r in audit if r['variant'] == variant]
            assert len(rows) == 1800
            events = Counter(float(r['latency_s']) for r in rows if request_pass(r))
            pooled_events[variant].update(events)
            for r in rows:
                key=(r['prompt_id'],r['run'])
                time=float(r['latency_s'])
                assert key not in paired_times or paired_times[key]==time
                paired_times[key]=time
                if request_pass(r):
                    for i,t in enumerate(BUDGETS):
                        pooled_groups[variant][r['prompt_id']][i] += int(time<=t)
            count = 0
            curves.append(dict(model=model, variant=variant, budget_s=0, count=0, n=1800, yield_pct=0))
            for time, n in sorted(events.items()):
                count += n
                curves.append(dict(model=model, variant=variant, budget_s=time,
                                   count=count, n=1800, yield_pct=pct(count,1800)))
            if variant != 'v3_public_surface_controller':
                continue
            grouped = defaultdict(list)
            for r in rows:
                pid = r['prompt_id']
                assert pid not in categories or categories[pid] == r['category']
                categories[pid] = r['category']
                grouped[pid].append(r)
            assert len(grouped) == 600 and all(len(v) == 3 for v in grouped.values())
            groups[model] = {pid: [qualified_yield(rs,t) for t in BUDGETS]
                             for pid,rs in grouped.items()}
            for t in BUDGETS:
                count = qualified_yield(rows,t)
                endpoints.append(dict(model=model,budget_s=t,count=count,n=1800,yield_pct=pct(count,1800)))
    samples = {model: [[] for _ in BUDGETS] for _,model in MODELS}
    policy_samples = {variant:[[] for _ in BUDGETS] for variant in POLICIES}
    for weights in bootstrap_weights(categories,bootstraps):
        for model, prompts in groups.items():
            sums = [0]*len(BUDGETS)
            for pid,weight in weights.items():
                for i,value in enumerate(prompts[pid]):
                    sums[i] += weight*value
            for i,value in enumerate(sums):
                samples[model][i].append(pct(value,1800))
        for variant in POLICIES:
            sums=[0]*len(BUDGETS)
            for pid,weight in weights.items():
                for i,value in enumerate(pooled_groups[variant][pid]):
                    sums[i] += weight*value
            for i,value in enumerate(sums):
                policy_samples[variant][i].append(pct(value,9000))
    for row in endpoints:
        vals = samples[row['model']][BUDGETS.index(row['budget_s'])]
        row.update(p025=quantile(vals,.025),p975=quantile(vals,.975))
    contrasts = []
    for i,t in enumerate(BUDGETS):
        vals = [q-g for q,g in zip(samples['Qwen2.5 7B'][i],samples['Gemma2 2B Q8'][i])]
        points = {r['model']:r['yield_pct'] for r in endpoints if r['budget_s']==t}
        contrasts.append(dict(comparison='Qwen minus Gemma2 2B Q8',budget_s=t,
                              difference_pp=points['Qwen2.5 7B']-points['Gemma2 2B Q8'],
                              p025=quantile(vals,.025),p975=quantile(vals,.975)))
    assert hashes == {p.name:sha256(p) for p in paths}
    policy_summary, policy_gap = [], []
    for variant in POLICIES:
        count=0
        curves.append(dict(model='ALL',variant=variant,budget_s=0,count=0,n=9000,yield_pct=0))
        for time,n in sorted(pooled_events[variant].items()):
            count+=n
            curves.append(dict(model='ALL',variant=variant,budget_s=time,count=count,n=9000,yield_pct=pct(count,9000)))
        for i,t in enumerate(BUDGETS):
            n=sum(value for time,value in pooled_events[variant].items() if time<=t)
            vals=policy_samples[variant][i]
            policy_summary.append(dict(variant=variant,budget_s=t,count=n,n=9000,yield_pct=pct(n,9000),
                                       p025=quantile(vals,.025),p975=quantile(vals,.975)))
    for i,t in enumerate(BUDGETS):
        values=[v3-v2 for v2,v3 in zip(policy_samples[POLICIES[0]][i],policy_samples[POLICIES[1]][i])]
        point={r['variant']:r['yield_pct'] for r in policy_summary if r['budget_s']==t}
        policy_gap.append(dict(budget_s=t,gain_pp=point[POLICIES[1]]-point[POLICIES[0]],
                               p025=quantile(values,.025),p975=quantile(values,.975)))
    outputs = {'latency_budget_curves':curves,'latency_budget_summary':endpoints,
               'latency_budget_contrasts':contrasts,'latency_budget_policy_summary':policy_summary,
               'latency_budget_policy_gap':policy_gap}
    for suffix,rows in outputs.items():
        write(results/f'pag_naacl_{suffix}.csv',rows)
    manifest = dict(model_generations=0,input_sha256=hashes,
                    code_sha256=sha256(Path(__file__)),budgets_s=BUDGETS,
                    bootstrap_replicates=bootstraps,bootstrap_seed=20261101,
                    numerator='Joint audit pass AND recorded generation latency <= budget.',
                    denominator='All 1800 observations per model, including failures and late responses.',
                    interval='2.5th/97.5th percentile; category-stratified unique prompts; all models/runs stay paired.',
                    scope='Generation round trip only; no renderer/network production SLO, confidence, or task-utility claim.',
                    outputs=list(outputs))
    (results/'pag_naacl_latency_budget_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'policy_summary':policy_summary,'policy_gap':policy_gap,'contrasts':contrasts},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--bootstraps',type=int,default=2000)
    args=parser.parse_args()
    if args.bootstraps<2:
        parser.error('--bootstraps must be at least 2')
    run(args.results_dir,args.bootstraps)
