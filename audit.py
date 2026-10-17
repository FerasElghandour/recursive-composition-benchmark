#!/usr/bin/env python3
"""Verify delivered artifacts and exactly repeat three representative seed-0 runs."""
import hashlib, json, tempfile
from pathlib import Path
import experiment as e
import numpy as np

def main():
    root=Path(__file__).parent;results=root/'results'
    e.verify()
    env=json.loads((results/'environment.json').read_text())
    assert env['source_sha256']==hashlib.sha256((root/'experiment.py').read_bytes()).hexdigest()
    for seed in range(5):
        stored=np.load(results/f'data_seed{seed}.npz');fresh=e.dataset(seed)
        assert set(stored.files)==set(fresh)
        for k,v in fresh.items():assert np.array_equal(stored[k],v),(seed,k)
        rr={m:json.loads((results/f'{m}_{seed}.json').read_text()) for m in e.METHODS}
        assert len({rr[m]['train_dense_macs'] for m in ('seed_balanced','seed_left','seed_random')})==1
        assert len({rr[m]['train_dense_macs'] for m in ('shared_balanced','shared_left','independent_capacity')})==1
        target=rr['shared_balanced']['train_dense_macs'];actual=rr['independent_compute']['train_dense_macs']
        assert target<=actual and (actual-target)/target<.001
        assert all(r['parameters']==2589 for m,r in rr.items() if m.startswith(('seed_','shared_')))
        for m,r in rr.items():
            assert r['parameters']>0 and r['parameter_bytes']==8*r['parameters']
            assert all(np.isfinite(v) for v in (r['train_cpu_seconds'],r['train_wall_seconds']))
            assert all(0<=s['exact']<=1 and 0<=s['state']<=1 for s in r['metrics'].values())
            if m.startswith('independent'):assert 'long' not in r['metrics']
    repeated=[]
    with tempfile.TemporaryDirectory(prefix='seed-recursive-audit-') as tmp:
        for method in ('seed_balanced','shared_balanced','modular_matrix'):
            r=e.run(0,method,3000,64,Path(tmp));old=json.loads((results/f'{method}_0.json').read_text())
            assert r['metrics']==old['metrics'] and r['history']==old['history']
            w1=np.load(Path(tmp)/f'weights_{method}_0.npz');w0=np.load(results/f'weights_{method}_0.npz')
            assert set(w1.files)==set(w0.files)
            for key in w1.files:assert np.array_equal(w1[key],w0[key]),(method,key)
            repeated.append({'method':method,'seed':0,'weights_bitwise_equal':True,'all_metrics_equal':True,'history_equal':True})
    verification={'gradient_target_split_and_cache_tests':'PASS','source_hash':'PASS','all_five_saved_datasets_equal_regeneration':True,'parameter_and_compute_budget_checks':'PASS','fresh_deterministic_reruns':repeated,'notes':'Exact reruns are verified in the recorded environment; times/RSS are not expected to repeat exactly.'}
    (results/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
