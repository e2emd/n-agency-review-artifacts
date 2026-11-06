"""Factorial renderer-component replay on the same saved V2/V3 generations."""
import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_runtime_v3 import parse_proposal, v2_public_surface, v3_public_surface_controller, classify
from naacl_deep_analysis import MODELS, ROOT, read, write, sha256, request_pass, pct

GATE_FIELDS = ['parse_ok','proposed_action','control_decision','public_action',
               'visible_control','visible_outcome','visible_final_line']
FLAGS = ['unsafe_public_surface','unsafe_final_line','unsafe_intent','raw_output_leak',
         'denied_nonrefusal','intent_visible_on_denied','over_refusal']


def component_surface(row, proposal, gate, metadata, raw):
    """G swaps the V2 gate; M redacts denied intent; S protects error/raw rendering."""
    surface = v2_public_surface(row,proposal)
    if gate:
        full = v3_public_surface_controller(row,proposal)
        surface.update({field:full[field] for field in GATE_FIELDS})
    if metadata and surface['control_decision']=='DENY':
        surface['visible_intent'] = ('[withheld: invalid structure]' if surface['parse_ok']=='False'
                                     else '[withheld by policy]')
    if raw:
        surface.update(visible_raw_output='',raw_output_visible='False',
                       visible_error_state=('error: invalid model structure withheld'
                                            if surface['parse_ok']=='False' else ''))
    return surface


def run(results):
    paths=[results/f'pag_runtime_v3_live_{slug}_full_{kind}.csv'
           for slug,_ in MODELS for kind in ['raw','audit']]
    hashes={p.name:sha256(p) for p in paths}
    groups=defaultdict(Counter)
    conditions=list(itertools.product([False,True],repeat=3))
    for slug,model in MODELS:
        raw=read(results/f'pag_runtime_v3_live_{slug}_full_raw.csv')
        audit=read(results/f'pag_runtime_v3_live_{slug}_full_audit.csv')
        saved={(r['prompt_id'],r['run'],r['variant']):r for r in audit}
        assert len(raw)==1800 and len(saved)==5400
        for row in raw:
            original=dict(row); proposal=parse_proposal(row)
            before=saved[(row['prompt_id'],row['run'],'v2_public_surface')]
            full=saved[(row['prompt_id'],row['run'],'v3_public_surface_controller')]
            for g,m,s in conditions:
                surface=component_surface(row,proposal,g,m,s)
                metrics=classify(row,surface)
                if (g,m,s)==(False,False,False):
                    assert all(before[k]==v for k,v in surface.items())
                    assert all(before[k]==v for k,v in metrics.items())
                if (g,m,s)==(True,True,True):
                    assert all(full[k]==v for k,v in surface.items())
                    assert all(full[k]==v for k,v in metrics.items())
                passed=request_pass({**surface,**metrics})
                base_pass=request_pass(before)
                count=Counter(n=1,n_denied=int(row['gold_permission']=='DENY'),
                              n_allowed=int(row['gold_permission']=='ALLOW'),audit_pass=int(passed),
                              newly_passing=int(passed and not base_pass),newly_failing=int(base_pass and not passed),
                              allowed_preserved=int(row['gold_permission']=='ALLOW' and
                                                    surface['visible_final_line']==before['visible_final_line']),
                              **{key:int(metrics[key]=='True') for key in FLAGS})
                condition=f'{int(g)}{int(m)}{int(s)}'
                for name in [model,'ALL']:groups[(name,condition)].update(count)
            assert row==original
    rows=[]
    for model in [name for _,name in MODELS]+['ALL']:
        for g,m,s in conditions:
            condition=f'{int(g)}{int(m)}{int(s)}';count=groups[(model,condition)]
            r=dict(model=model,condition=condition,G=int(g),M=int(m),S=int(s),**count)
            r['audit_pass_pct']=pct(count['audit_pass'],count['n'])
            r['gain_vs_v2_pp']=pct(count['audit_pass']-groups[(model,'000')]['audit_pass'],count['n'])
            r['allowed_preservation_pct']=pct(count['allowed_preserved'],count['n_allowed'])
            for key in FLAGS:
                denominator=count['n_denied'] if key in ['denied_nonrefusal','intent_visible_on_denied'] else (
                    count['n_allowed'] if key=='over_refusal' else count['n'])
                r[key+'_pct']=pct(count[key],denominator)
            rows.append(r)
    assert hashes=={p.name:sha256(p) for p in paths}
    write(results/'pag_naacl_component_ablation.csv',rows)
    manifest=dict(model_generations=0,input_sha256=hashes,code_sha256=sha256(Path(__file__)),
                  component_order=['G','M','S'],conditions=[r['condition'] for r in rows if r['model']=='ALL'],
                  G='Replace the saved V2 action/generic-line gate with the V3 annotation-conditioned line+intent gate.',
                  M='Redact intent when the selected gate denies; does not alter the final line.',
                  S='Hide raw output and use generic safe error text on parse failure; does not alter the gate.',
                  invariant='G=M=S=0 exactly reproduces V2; G=M=S=1 exactly reproduces V3; scoring always uses original annotations.',
                  scope='Fixed-output renderer ablation, not a new live generation run or retraining experiment.')
    (results/'pag_naacl_component_ablation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps([r for r in rows if r['model']=='ALL'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
    run(parser.parse_args().results_dir)
