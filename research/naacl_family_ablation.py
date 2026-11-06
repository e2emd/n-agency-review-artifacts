"""Compare generation/policy families and paired schema-controller ablations."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_runtime_v3 import classify, parse_proposal, v2_public_surface, v3_public_surface_controller
from reanalyze_runtime_v2 import apply_runtime_v2
from naacl_deep_analysis import MODELS, ROOT, METRICS, read, write, sha256, request_pass, pct

VARIANTS = [
    ('safety_prompt', 'safety_prompt', 'native'),
    ('schema_only', 'schema_only', 'native'),
    ('self_policy', 'schema_policy', 'native'),
    ('schema_v2', 'schema_only', 'v2'),
    ('schema_v3', 'schema_only', 'v3'),
]


def surface_for(row, controller):
    """Native views preserve original decisions; V2/V3 share a fixed proposal."""
    if any(k.endswith('_v2') for k in row):
        raise ValueError('Use original generation rows, not pre-controlled views')
    proposal = parse_proposal(row)
    if controller == 'native':
        return v2_public_surface(row, proposal)
    if controller == 'v2':
        controlled = {**row, **apply_runtime_v2(row)}
        return v2_public_surface(controlled, proposal)
    if controller == 'v3':
        return v3_public_surface_controller(row, proposal)
    raise ValueError(controller)


def run(results):
    paths = [results / f'pag_eval_llamacpp_{slug}{suffix}.csv'
             for slug, _ in MODELS for suffix in ['_full', '_schema_policy_full']]
    paths += [results / 'pag_runtime_v3_public_surface_audit.csv']
    hashes = {p.name: sha256(p) for p in paths}
    reference = {(r['model'], r['prompt_id'], r['run'], r['variant']): r
                 for r in read(paths[-1]) if r['condition'] == 'schema_only'}
    groups = defaultdict(Counter)
    matched = 0
    for slug, model in MODELS:
        source = read(results / f'pag_eval_llamacpp_{slug}_full.csv')
        source += read(results / f'pag_eval_llamacpp_{slug}_schema_policy_full.csv')
        by_condition = {c: [r for r in source if r['condition'] == c]
                        for c in ['safety_prompt', 'schema_only', 'schema_policy']}
        keys = [{(r['prompt_id'], r['run']) for r in rows} for rows in by_condition.values()]
        assert all(len(rows) == 1800 for rows in by_condition.values())
        assert all(k == keys[0] and len(k) == 1800 for k in keys)
        prompts = [{r['prompt_id']: (r['prompt'], r['category'], r['gold_permission'])
                    for r in rows} for rows in by_condition.values()]
        assert all(p == prompts[0] and len(p) == 600 for p in prompts)
        for variant, condition, controller in VARIANTS:
            for row in by_condition[condition]:
                unchanged = dict(row)
                surface = surface_for(row, controller)
                metrics = classify(row, surface)
                if controller != 'native':
                    name = 'v2_public_surface' if controller == 'v2' else 'v3_public_surface_controller'
                    saved = reference[(row['model'], row['prompt_id'], row['run'], name)]
                    assert all(saved[k] == v for k, v in surface.items())
                    assert all(saved[k] == v for k, v in metrics.items())
                    matched += 1
                baseline = surface_for(row, 'native')
                baseline_pass = request_pass({**baseline, **classify(row, baseline)})
                passed = request_pass({**surface, **metrics})
                v2_surface = surface_for(row, 'v2') if controller == 'v3' else surface
                v2_pass = request_pass({**v2_surface, **classify(row, v2_surface)})
                counts = Counter(n=1, n_denied=int(row['gold_permission'] == 'DENY'),
                                 n_allowed=int(row['gold_permission'] == 'ALLOW'),
                                 latency_total_s=float(row['latency_s']),
                                 audit_pass=int(passed),
                                 newly_passing=int(passed and not baseline_pass),
                                 newly_failing=int(baseline_pass and not passed),
                                 newly_passing_vs_v2=int(controller == 'v3' and passed and not v2_pass),
                                 newly_failing_vs_v2=int(controller == 'v3' and v2_pass and not passed),
                                 allowed_preserved=int(row['gold_permission'] == 'ALLOW' and
                                                       surface['visible_final_line'] == baseline['visible_final_line']),
                                 **{k: int(metrics[k] == 'True') for k in METRICS})
                for key in [(model, variant), ('ALL', variant)]:
                    groups[key].update(counts)
                assert row == unchanged
    output = []
    for model in [m for _, m in MODELS] + ['ALL']:
        for variant, condition, controller in VARIANTS:
            count = groups[(model, variant)]
            row = dict(model=model, variant=variant, generation_condition=condition,
                       controller=controller, **count)
            row['audit_pass_pct'] = pct(count['audit_pass'], count['n'])
            row['latency_mean_s'] = count['latency_total_s'] / count['n']
            row['allowed_preservation_pct'] = pct(count['allowed_preserved'], count['n_allowed'])
            for key in METRICS:
                denom = count['n_denied'] if key in ['denied_nonrefusal', 'intent_visible_on_denied'] else (
                    count['n_allowed'] if key == 'over_refusal' else count['n'])
                row[key + '_pct'] = pct(count[key], denom)
            output.append(row)
    assert matched == 18000
    assert groups[('ALL', 'schema_v3')]['newly_passing_vs_v2'] == 5250
    assert groups[('ALL', 'schema_v3')]['newly_failing_vs_v2'] == 0
    assert len({groups[('ALL', v)]['latency_total_s'] for v in
                ['schema_only', 'schema_v2', 'schema_v3']}) == 1
    assert hashes == {p.name: sha256(p) for p in paths}
    write(results / 'pag_naacl_family_ablation.csv', output)
    manifest = dict(model_generations=0, input_sha256=hashes,
                    code_sha256=sha256(Path(__file__)), matched_saved_views=matched,
                    unique_prompts=600, observations_per_variant=9000,
                    source_generation_records=27000,
                    latency='Mean recorded generation round-trip seconds; Schema rows share times; controller/rendering processing is excluded.',
                    paired_variants=['schema_only', 'schema_v2', 'schema_v3'],
                    scoring='Original annotations and the same full-surface detector for every condition.',
                    native_contract='Original final/control/action; structured intent shown; raw visible on parse failure. Text-only safety has no intent field.',
                    scope='Generation-family comparisons are different prompts/contracts; only the three schema rows hold proposals fixed. V3 uses the annotation-conditioned detector.')
    (results / 'pag_naacl_family_ablation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps([r for r in output if r['model'] == 'ALL'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    run(parser.parse_args().results_dir)
