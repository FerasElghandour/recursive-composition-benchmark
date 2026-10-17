#!/usr/bin/env python3
"""Regenerate tables, diagnostics, report and an SVG from completed runs."""
import argparse, collections, hashlib, json, math, statistics
from pathlib import Path
import numpy as np
from experiment import METHODS, Neural, Matrix, dataset, target, PERMS, tree, count
LABELS={'seed_balanced':'Seed + balanced stages','seed_left':'Seed + left stages','seed_random':'Seed + random stages','shared_balanced':'Shared balanced, joint','shared_left':'Shared left, joint','independent':'Independent, total params matched','independent_compute':'Independent, compute matched','independent_capacity':'Independent, per-task width matched*','modular_matrix':'Modular matrix (known algebra)'}

def mean(a):return float(np.mean(a))
def sd(a):return float(np.std(a,ddof=1)) if len(a)>1 else 0.
def fmt(a,scale=1,digits=1):return f'{mean(a)*scale:.{digits}f} ± {sd(a)*scale:.{digits}f}'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path(__file__).parent/'results');a=ap.parse_args();root=a.results.parent
    methods=[m for m in METHODS if (a.results/f'{m}_0.json').exists()]
    runs={m:[json.loads(p.read_text()) for p in sorted(a.results.glob(f'{m}_[0-9]*.json'))] for m in methods}
    assert all(len(rr)==5 for rr in runs.values())
    stats={}
    for m,rr in runs.items():
        stats[m]={}
        for metric in ('train','iid','ood','long'):
            xs=[r['metrics'][metric]['exact'] for r in rr if metric in r['metrics']]
            if xs:stats[m][metric]={'mean':mean(xs),'sd':sd(xs),'min':min(xs),'max':max(xs),'seeds':xs}
    # Paired topology/curriculum comparisons; no inferential significance claim.
    pairs={}
    for m,n in [('seed_balanced','seed_left'),('seed_balanced','seed_random'),('seed_balanced','shared_balanced'),('seed_left','shared_left'),('seed_balanced','independent_compute'),('seed_left','independent_capacity')]:
        if m not in stats or n not in stats:continue
        dif=np.array(stats[m]['ood']['seeds'])-np.array(stats[n]['ood']['seeds'])
        pairs[f'{m} minus {n}']={'mean_pp':mean(dif)*100,'sd_pp':sd(dif)*100,'per_seed_pp':(100*dif).tolist()}
    # Fixed training-frequency majority output map, no test-label fitting.
    refs=[];sizes={};diagnostics={}
    for seed in range(5):
        d=dataset(seed);sizes[str(seed)]={k:len(v) for k,v in d.items()}
        freq=collections.defaultdict(float)
        for n in range(1,7):
            y=target(d[f'train_{n}'])
            for row in y:freq[tuple(row)]+=1/(6*len(y))
        majority=max(freq,key=freq.get)
        refs.append({s:float(np.concatenate([np.all(target(x)==majority,axis=1) for k,x in d.items() if k.startswith(s+'_') and len(x)]).mean()) for s in ('iid','ood','long')})
    for m,rr in runs.items():
        ds=[]
        for r in rr:
            seed=r['seed'];d=dataset(seed);weights=np.load(a.results/f'weights_{m}_{seed}.npz');ind=m.startswith('independent')
            models=[Neural(31 if m=='independent_capacity' else 10,seed*100+13+i,'balanced') for i in range(6)] if ind else ([Matrix(seed*100+13)] if m=='modular_matrix' else [Neural(31,seed*100+13,'random' if m=='seed_random' else ('left' if m.endswith('left') else 'balanced'))])
            for i,model in enumerate(models):
                for k in model.p:model.p[k][...]=weights[f'{i}_{k}']
            invalid=[];correct=[];failures=[]
            for n in range(2,7):
                x=d[f'ood_{n}'];model=models[n-1] if ind else models[0];pred=model.predict(x);y=target(x)
                invalid.extend(np.any(np.sort(pred,axis=1)!=np.arange(4),axis=1).tolist())
                correct.extend(np.all(pred==y,axis=1).tolist())
                for i in np.where(np.any(pred!=y,axis=1))[0][:1]:failures.append({'length':n,'program':x[i].tolist(),'target':y[i].tolist(),'predicted':pred[i].tolist()})
            assert math.isclose(mean(correct),r['metrics']['ood']['exact'],abs_tol=1e-14)
            ds.append({'seed':seed,'invalid_ood_map_fraction':mean(invalid),'failures':failures})
        diagnostics[m]=ds
    summary={'stats':stats,'paired':pairs,'majority_reference':refs,'sizes':sizes,'diagnostics':diagnostics}
    (a.results/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Seed-Based Recursive Intelligence: completed small experiment','',
    'Run date: 15 October 2026. Five seeds (0–4), CPU, NumPy float64. Values are mean ± sample standard deviation across seeds. All final checkpoints are reported; no seed was excluded.','',
    '## Finding','',
    'The balanced addition-chain rule did **not** improve this implementation. Left-to-right recursive composition performed better on held-out combinations in every seed. A standard modular model supplied with the correct composition algebra solved every evaluated program. These results do not establish emergence, novelty, or a new intelligence architecture.','',
    '## Accuracy','',
    'Exact accuracy requires the entire four-state map to be correct. The primary OOD split contains previously forbidden adjacent command pairs. IID means previously unseen programs from the permitted grammar. Long tests use permitted pairs at lengths 8 and 12.','',
    '| Model | Training % | IID % | Held-out pairs % | Long % |','|---|---:|---:|---:|---:|']
    for m in methods:
        cells=[fmt([r['metrics'][s]['exact'] for r in runs[m]],100) if s in runs[m][0]['metrics'] else 'N/A' for s in ('train','iid','ood','long')]
        lines.append('| '+LABELS[m]+' | '+' | '.join(cells)+' |')
    lines+=['','Independent models have no trained network for lengths 8/12; they are not assigned a zero score. *The per-task-width comparison is an explicitly post hoc sensitivity check; see PROTOCOL_AMENDMENT.md.','',
    f"A predictor that always emits the most frequent training map scores {fmt([r['ood'] for r in refs],100)}% on held-out pairs. This is an empirical reference, not a universal chance rate.",'',
    '## Compute, time and memory','',
    'The neural topology comparison has identical width, training minibatches and dense MAC counts. Joint and staged curricula have the same number of examples, but different length distributions and therefore different computation. MACs count dense forward/backward matrix multiplications only; approximate FLOPs = 2 × MACs. Elementwise operations, Adam, embedding scatter and progress evaluation are excluded from MACs. Times include six progress evaluations (more for the extended independent run).','',
    '| Model | Parameters | Updates | Examples | Train G-MAC | Wall s | CPU s | Peak RSS MiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        rr=runs[m]
        lines.append(f"| {LABELS[m]} | {rr[0]['parameters']:,} | {mean([r['updates'] for r in rr]):,.0f} | {mean([r['examples'] for r in rr]):,.0f} | {fmt([r['train_dense_macs'] for r in rr],1e-9,3)} | {fmt([r['train_wall_seconds'] for r in rr],1,3)} | {fmt([r['train_cpu_seconds'] for r in rr],1,3)} | {fmt([r['peak_rss_bytes'] for r in rr],1/2**20,2)} |")
    lines+=['','Each run used a fresh serial worker process. Peak RSS includes Python, NumPy, generated data and evaluation, so it is not pure model or training memory. Short runs have noisy timing and startup/data-generation time is excluded from training seconds. The independent compute run reaches the shared joint balanced budget within at most one update; its smaller matrices can take longer despite equal dense MACs.','',
    '| Model | Weights KiB | Weights + gradients + Adam KiB | Inference on 512 programs ms | Cache payload KiB | Cached inference ms |','|---|---:|---:|---:|---:|---:|']
    for m in methods:
        rr=runs[m];mem=[r['memoization'] for r in rr if r['memoization']]
        lines.append(f"| {LABELS[m]} | {rr[0]['parameter_bytes']/1024:.2f} | {rr[0]['parameter_grad_adam_bytes']/1024:.2f} | {fmt([r['inference_512_seconds'] for r in rr],1000,3)} | {fmt([r['array_bytes'] for r in mem],1/1024,2) if mem else 'N/A'} | {fmt([r['seconds'] for r in mem],1000,3) if mem else 'N/A'} |")
    lines+=['','Inference is on the 512 length-6 held-out-pair programs. Uncached timing is the mean of five batched calls; cached timing is one cold-cache, program-by-program traversal. These implementations have different batching overheads, so this does not isolate an algorithmic caching speedup. Cache payload excludes keys, dictionaries and Python object overhead. No cache survives a training parameter update. Working arrays and temporary optimizer allocations are additional to the persistent tensor counts.','',
    '## Paired comparisons','',
    'Held-out-pair accuracy differences in percentage points, pairing the same seed and split. With only five seeds these are descriptive results; no population significance or broad superiority is claimed.','',
    '| First minus second | Mean ± SD pp | Per-seed pp |','|---|---:|---|']
    for key,v in pairs.items():lines.append(f"| {key} | {v['mean_pp']:.2f} ± {v['sd_pp']:.2f} | {', '.join(f'{x:.2f}' for x in v['per_seed_pp'])} |")
    lines+=['','## Failures and retention','',
    '| Model | OOD state accuracy % | Invalid OOD map % | Final primitive accuracy % | Primitive change from first checkpoint pp |','|---|---:|---:|---:|---:|']
    for m in methods:
        rr=runs[m]
        lines.append(f"| {LABELS[m]} | {fmt([r['metrics']['ood']['state'] for r in rr],100)} | {fmt([r['invalid_ood_map_fraction'] for r in diagnostics[m]],100)} | {fmt([r['metrics']['train_1']['exact'] for r in rr],100)} | {fmt([r['primitive_change_since_first_checkpoint'] for r in rr],100)} |")
    lines+=['','Invalid means the prediction repeats an output state and therefore cannot be a permutation. The neural decoder does not enforce validity. The modular baseline always predicts valid maps in the observed tests. Primitive retention covers four elementary commands only, with replay; it is not a broad continual-learning evaluation. Raw JSON contains per-length accuracy, training checkpoints and specific failures.','',
    'Examples from seed 0 (zero-based command/state IDs):','',
    '| Model | Program | Correct map | Predicted map |','|---|---|---|---|']
    for m in methods:
        f=diagnostics[m][0]['failures']
        if f:
            v=f[0];lines.append(f"| {LABELS[m]} | {v['program']} | {v['target']} | {v['predicted']} |")
    lines+=['','## Interpretation and limits','',
    '1. **Balance did not help here.** Balanced assembly lost to left assembly under the same staged training and same dense computation. Their binary-tree node counts are identical. A balanced tree has shorter dependency depth, but this implementation executes tree nodes serially; it does not benchmark parallel hardware scheduling.','2. **Reusing learned components is viable, but the numerical rule is not established as useful.** Sharing plus an appropriate composition operator can generalize. The matrix model is advantaged by knowing associativity and matrix multiplication; it learns only primitive meanings. It is not parameter- or optimizer-matched and should not be treated as evidence that all modular models will solve harder tasks.','3. **Poor optimization constrains the negative result.** Balanced models fail to fit substantial portions of their training set; this cannot distinguish an inherently poor rule from an unsuitable tanh composer, width or optimizer budget. A plausible explanation for the left advantage is that the composer usually combines a running result with a primitive, while balanced assembly asks it to combine two larger latent programs. This mechanism was not directly tested.','4. **The seed is an encoder template, not a magical constant.** The four commands use one learned leaf-encoding table. Recursion reuses that table and the same composer. Representations become larger programs, while the number of trainable parameters remains fixed; this is not autonomous network or skill growth.','5. **Independent learning is defined by length.** Matching total parameters makes each independent task model narrower. The extended run matches approximate dense compute but sees many more repeated examples. The post hoc width check matches per-task capacity at six times aggregate parameters. No single comparison simultaneously matches total parameters, per-task width, data and runtime.','6. **The benchmark is small and structured.** Four states, four fixed permutations, two held-out directed pairs, one underlying algebra and short programs. A legal training program can share its output map with a forbidden test program; the split withholds syntax combinations, not output classes. Results do not establish performance on language, perception, real-world continual learning or other algebras.','7. **No data-efficiency curve was run.** Fixed-budget comparisons and repeated-example counts do not establish sample complexity or that a target accuracy can be reached with less data. There is no Transformer/Tree-LSTM comparison or tuned published system replication.','8. **Small seed count and timing scope.** Five seeds vary initialization, minibatches and program subsets together; they do not independently estimate each source of variation. CPU runtime is one environment and analytic MACs are only a proxy. All arithmetic is float64. Hardware/BLAS versions can affect times and low-order numeric results.','',
    'A useful next experiment would keep the same held-out split, vary training-set size and compute budgets, and compare stronger associative/neural composers on several independent transformation algebras. That would test whether these findings survive removal of this experiment’s main bottlenecks.','',
    '## Research context','',
    'Recursive neural composition predates this experiment; see [Socher et al., 2013](https://aclanthology.org/D13-1170/). Holding out systematic command combinations is motivated by the generalization question studied in [Lake & Baroni, 2018](https://proceedings.mlr.press/v80/lake18a.html). This benchmark is newly generated for this small test and is **not SCAN**. The implemented baselines are controlled elementary instances, not reproductions of those papers.','',
    'See README.md for the full data-generation description, executable commands and file manifest. PROTOCOL.md records the original fixed design; PROTOCOL_AMENDMENT.md records the post hoc capacity check.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for m in methods:print(m,fmt([r['metrics']['ood']['exact'] for r in runs[m]],100))
    print('Generated REPORT.md and results/summary.json')
if __name__=='__main__':main()
