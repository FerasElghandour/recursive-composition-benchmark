#!/usr/bin/env python3
"""Dependency-light controlled compositional learning experiment. See README.md."""
import os
for _v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse, csv, hashlib, itertools, json, platform, resource, subprocess, sys, time
from pathlib import Path
import numpy as np

K=4
PERMS=np.array([[1,2,3,0],[1,0,2,3],[0,2,1,3],[3,2,1,0]],dtype=np.int64)
FORBIDDEN={(0,1),(2,3)}
METHODS=['seed_balanced','seed_left','seed_random','shared_balanced','shared_left',
         'independent','independent_compute','independent_capacity','modular_matrix']

def target(x):
    y=np.tile(np.arange(K),(len(x),1))
    for j in range(x.shape[1]): y=PERMS[x[:,j,None],y]
    return y

def forbidden(p): return any(tuple(p[i:i+2]) in FORBIDDEN for i in range(len(p)-1))

def dataset(seed):
    rng=np.random.default_rng(10000+seed)
    d={}
    for n in range(1,7):
        allp=list(itertools.product(range(4),repeat=n))
        seen=[p for p in allp if not forbidden(p)]
        novel=[p for p in allp if forbidden(p)]
        rng.shuffle(seen); rng.shuffle(novel)
        # Preserve some same-grammar programs for an IID combination test where possible.
        nt=len(seen) if n<=2 else min(512,max(1,int(.8*len(seen))))
        d[f'train_{n}']=np.array(seen[:nt],dtype=np.int64).reshape(-1,n)
        d[f'iid_{n}']=np.array(seen[nt:nt+256],dtype=np.int64).reshape(-1,n)
        d[f'ood_{n}']=np.array(novel[:512],dtype=np.int64).reshape(-1,n)
    for n in (8,12):
        # Long-length tests contain no forbidden pairs, isolating length shift.
        ps=set()
        while len(ps)<512:
            p=tuple(rng.integers(4,size=n).tolist())
            if not forbidden(p): ps.add(p)
        d[f'long_{n}']=np.array(sorted(ps),dtype=np.int64)
    for n in range(1,7):
        train=set(map(tuple,d[f'train_{n}']))
        assert all(not forbidden(p) for p in train)
        for split in ('iid','ood'):
            test=set(map(tuple,d[f'{split}_{n}']))
            assert not train.intersection(test)
            if split=='ood': assert all(forbidden(p) for p in test)
    return d

def tree(n,kind,rng):
    if n==1:return 0
    q=n//2 if kind=='balanced' else (n-1 if kind=='left' else int(rng.integers(1,n)))
    # Index leaves by their position after recursive construction.
    def build(lo,hi):
        m=hi-lo
        if m==1:return lo
        k=m//2 if kind=='balanced' else (m-1 if kind=='left' else int(rng.integers(1,m)))
        return (build(lo,lo+k),build(lo+k,hi))
    return build(0,n)

class Adam:
    def __init__(self,p):self.m={k:np.zeros_like(v) for k,v in p.items()};self.v={k:np.zeros_like(v) for k,v in p.items()};self.t=0
    def step(self,p,g,lr):
        self.t+=1
        norm=np.sqrt(sum(float(np.sum(a*a)) for a in g.values()))
        scale=min(1.,5./max(norm,1e-12))
        for k in p:
            a=g[k]*scale;self.m[k]=.9*self.m[k]+.1*a;self.v[k]=.999*self.v[k]+.001*a*a
            p[k]-=lr*(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)

def softmax(z):
    e=np.exp(z-z.max(axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)

class Neural:
    def __init__(self,h,seed,kind):
        r=np.random.default_rng(seed);self.h=h;self.kind=kind
        self.p={'E':r.normal(0,.4,(4,h)),'W':r.normal(0,1/np.sqrt(2*h),(2*h,h)),
                'b':np.zeros(h),'D':r.normal(0,1/np.sqrt(h),(h,K*K)),'c':np.zeros(K*K)}
        self.opt=Adam(self.p)
        self.trees={n:tree(n,kind,np.random.default_rng(seed+7907*n)) for n in range(1,13)}
    def loss(self,x,y,grad=True):
        p=self.p;cache=[]
        def rec(t):
            if isinstance(t,int):
                z=p['E'][x[:,t]];cache.append(('leaf',t,z));return len(cache)-1
            a,b=rec(t[0]),rec(t[1]);cat=np.concatenate([cache[a][-1],cache[b][-1]],axis=1)
            z=np.tanh(cat@p['W']+p['b']);cache.append(('node',a,b,cat,z));return len(cache)-1
        root=rec(self.trees[x.shape[1]]);h=cache[root][-1]
        probs=softmax((h@p['D']+p['c']).reshape(-1,K,K))
        loss=-np.log(np.maximum(probs[np.arange(len(x))[:,None],np.arange(K),y],1e-300)).mean()
        if not grad:return loss,probs.argmax(-1)
        dl=probs.copy();dl[np.arange(len(x))[:,None],np.arange(K),y]-=1;dl=dl.reshape(-1,K*K)/(len(x)*K)
        g={k:np.zeros_like(v) for k,v in p.items()};g['D']=h.T@dl;g['c']=dl.sum(0)
        dh={root:dl@p['D'].T}
        for i in range(len(cache)-1,-1,-1):
            c=cache[i];up=dh[i]
            if c[0]=='leaf':np.add.at(g['E'],x[:,c[1]],up)
            else:
                _,a,b,cat,z=c;dz=up*(1-z*z);g['W']+=cat.T@dz;g['b']+=dz.sum(0)
                dc=dz@p['W'].T;dh[a]=dc[:,:self.h];dh[b]=dc[:,self.h:]
        return loss,g
    def predict(self,x):return self.loss(x,target(x),False)[1]
    def forward_macs(self,n,b):return b*((n-1)*2*self.h*self.h+self.h*K*K)
    def memo_predict(self,x):
        cache={};hits=0
        def rec(p,t):
            nonlocal hits
            key=(p,str(t)) # Relative subtree shape distinguishes random layouts.
            if key in cache:hits+=1;return cache[key]
            if isinstance(t,int):v=self.p['E'][p[0]].copy()
            else:
                # Trees are renumbered to relative indices for cache reuse.
                nl=count(t[0]);a=rec(p[:nl],renumber(t[0]));b=rec(p[nl:],renumber(t[1]))
                v=np.tanh(np.concatenate([a,b])@self.p['W']+self.p['b'])
            cache[key]=v;return v
        pred=[]
        for row in x:
            h=rec(tuple(row.tolist()),self.trees[len(row)])
            pred.append((h@self.p['D']+self.p['c']).reshape(K,K).argmax(-1))
        return np.array(pred),sum(v.nbytes for v in cache.values()),len(cache),hits

def count(t):return 1 if isinstance(t,int) else count(t[0])+count(t[1])
def renumber(t):
    it=iter(range(count(t)))
    def rec(t):return next(it) if isinstance(t,int) else (rec(t[0]),rec(t[1]))
    return rec(t)

class Matrix:
    """Standard modular baseline: learned primitive matrices; known associative product."""
    def __init__(self,seed):
        self.p={'A':np.random.default_rng(seed).normal(0,.1,(4,K,K))};self.opt=Adam(self.p)
    def loss(self,x,y,grad=True):
        a=softmax(self.p['A']);s=np.tile(np.eye(K),(len(x),1,1));ss=[s]
        for j in range(x.shape[1]):s=s@a[x[:,j]];ss.append(s)
        loss=-np.log(np.maximum(s[np.arange(len(x))[:,None],np.arange(K),y],1e-300)).mean()
        if not grad:return loss,s.argmax(-1)
        ds=np.zeros_like(s);ds[np.arange(len(x))[:,None],np.arange(K),y]=-1/np.maximum(s[np.arange(len(x))[:,None],np.arange(K),y],1e-300)/(len(x)*K)
        da=np.zeros_like(a)
        for j in range(x.shape[1]-1,-1,-1):
            np.add.at(da,x[:,j],ss[j].transpose(0,2,1)@ds)
            ds=ds@a[x[:,j]].transpose(0,2,1)
        return loss,{'A':a*(da-(da*a).sum(-1,keepdims=True))}
    def predict(self,x):return self.loss(x,target(x),False)[1]
    def forward_macs(self,n,b):return b*n*K**3


def evaluate(models,d,independent):
    out={};fails=[]
    for key,x in d.items():
        if not len(x):continue
        n=x.shape[1]
        if independent and n>6:continue
        m=models[n-1] if independent else models[0]
        y=target(x);pred=m.predict(x);correct=(pred==y)
        out[key]={'n':len(x),'exact':float(correct.all(1).mean()),'state':float(correct.mean())}
        if key.startswith(('ood','long')):
            for i in np.where(~correct.all(1))[0][:2]:
                fails.append({'split':key,'program':x[i].tolist(),'target':y[i].tolist(),'predicted':pred[i].tolist()})
    for prefix in ('train','iid','ood','long'):
        a=[v for k,v in out.items() if k.startswith(prefix+'_')]
        if a:out[prefix]={'n':sum(v['n'] for v in a),'exact':sum(v['n']*v['exact'] for v in a)/sum(v['n'] for v in a),'state':sum(v['n']*v['state'] for v in a)/sum(v['n'] for v in a)}
    return out,fails


def run(seed,method,steps,batch,outdir):
    d=dataset(seed);np.savez_compressed(outdir/f'data_seed{seed}.npz',**d)
    independent=method.startswith('independent');structured=method=='modular_matrix'
    kind='random' if method=='seed_random' else ('left' if method.endswith('left') else 'balanced')
    models=[Neural(31 if method=='independent_capacity' else 10,seed*100+13+i,kind) for i in range(6)] if independent else ([Matrix(seed*100+13)] if structured else [Neural(31,seed*100+13,kind)])
    # Joint conditions share one sampled minibatch stream; staged conditions share another.
    rng=np.random.default_rng(20000+seed);hist=[];macs=0;examples=0;updates=0;stage1=None
    wall=time.perf_counter();cpu=time.process_time()
    budget=None
    if method=='independent_compute':
        # Replay same joint stream as shared_balanced to obtain its exact dense-MAC budget.
        ref=np.random.default_rng(20000+seed);budget=0
        for _ in range(steps):
            n=int(ref.integers(1,7));ref.integers(len(d[f'train_{n}']),size=batch)
            budget+=3*batch*((n-1)*2*31*31+31*K*K)
    while updates<steps or (budget is not None and macs<budget):
        if method.startswith('seed_'):
            stage=min(6,updates//(steps//6)+1)
            # Replay one third of updates on earlier lengths, retaining seed knowledge.
            n=stage if stage==1 or rng.random()<2/3 else int(rng.integers(1,stage))
        else:n=int(rng.integers(1,7))
        x=d[f'train_{n}'];ix=rng.integers(len(x),size=batch);xb=x[ix];yb=target(xb)
        m=models[n-1] if independent else models[0]
        loss,g=m.loss(xb,yb)
        if not np.isfinite(loss) or not all(np.isfinite(v).all() for v in g.values()):raise FloatingPointError((method,seed,updates))
        m.opt.step(m.p,g,.003 if not structured else .03)
        macs+=3*m.forward_macs(n,batch);examples+=batch;updates+=1
        if updates%max(1,steps//6)==0:
            ev,_=evaluate(models,{k:v for k,v in d.items() if k.startswith('train')},independent)
            hist.append({'update':updates,'loss':float(loss),'primitive_accuracy':ev['train_1']['exact'],'train_accuracy':ev['train']['exact']})
            if stage1 is None:stage1=ev['train_1']['exact']
    training_wall=time.perf_counter()-wall;training_cpu=time.process_time()-cpu
    ev,fails=evaluate(models,d,independent)
    params=sum(v.size for m in models for v in m.p.values());pbytes=sum(v.nbytes for m in models for v in m.p.values())
    x=d['ood_6'];wall=time.perf_counter()
    for _ in range(5):pred=(models[5] if independent else models[0]).predict(x)
    inference=(time.perf_counter()-wall)/5
    memo=None
    if not structured:
        m=models[5] if independent else models[0];wall=time.perf_counter();mp,cb,cn,ch=m.memo_predict(x);mt=time.perf_counter()-wall
        assert np.array_equal(mp,pred),'Cache changed predictions'
        memo={'seconds':mt,'array_bytes':cb,'nodes':cn,'hits':ch,'python_object_overhead_included':False}
    # RSS is the isolated worker process peak, including NumPy, data and evaluation.
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes=rss if sys.platform=='darwin' else rss*1024
    r={'seed':seed,'method':method,'parameters':params,'parameter_bytes':pbytes,
       'parameter_plus_adam_bytes':3*pbytes,'parameter_grad_adam_bytes':4*pbytes,'peak_rss_bytes':rss_bytes,
       'updates':updates,'examples':examples,'train_dense_macs':macs,'approx_dense_flops':2*macs,
       'train_wall_seconds':training_wall,'train_cpu_seconds':training_cpu,'inference_512_seconds':inference,
       'memoization':memo,'metrics':ev,'history':hist,'failures':fails,
       'primitive_change_since_first_checkpoint':ev['train_1']['exact']-stage1}
    # Save learned parameters for audit / inference without training again.
    np.savez_compressed(outdir/f'weights_{method}_{seed}.npz',**{f'{i}_{k}':v for i,m in enumerate(models) for k,v in m.p.items()})
    (outdir/f'{method}_{seed}.json').write_text(json.dumps(r,indent=2))
    return r


def verify():
    d=dataset(0)
    for kind in ('balanced','left','random'):
        m=Neural(5,7,kind);x=d['train_4'][:3];y=target(x);loss,g=m.loss(x,y)
        for k in m.p:
            for idx in list(np.ndindex(m.p[k].shape))[:8]:
                orig=m.p[k][idx];eps=1e-5;m.p[k][idx]=orig+eps;l1=m.loss(x,y,False)[0]
                m.p[k][idx]=orig-eps;l0=m.loss(x,y,False)[0];m.p[k][idx]=orig
                assert np.isclose((l1-l0)/(2*eps),g[k][idx],atol=1e-6,rtol=1e-4),(kind,k,idx)
        assert np.array_equal(m.predict(x),m.memo_predict(x)[0])
    m=Matrix(7);x=d['train_4'][:3];y=target(x);_,g=m.loss(x,y)
    for idx in np.ndindex(m.p['A'].shape):
        orig=m.p['A'][idx];eps=1e-5;m.p['A'][idx]=orig+eps;l1=m.loss(x,y,False)[0]
        m.p['A'][idx]=orig-eps;l0=m.loss(x,y,False)[0];m.p['A'][idx]=orig
        assert np.isclose((l1-l0)/(2*eps),g['A'][idx],atol=1e-6,rtol=1e-4)
    # Explicit noncommutativity and an independent symbolic reference.
    assert not np.array_equal(target(np.array([[0,1]])),target(np.array([[1,0]])))
    for p in itertools.product(range(4),repeat=4):
        expected=[]
        for s in range(K):
            for t in p:s=int(PERMS[t,s])
            expected.append(s)
        assert target(np.array([p])).tolist()[0]==expected
    print('PASS: finite-difference gradients (all architectures), splits, targets, cache equivalence',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    ap.add_argument('--seeds',default='0,1,2,3,4');ap.add_argument('--methods',default=','.join(METHODS));ap.add_argument('--steps',type=int,default=3000);ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--worker',action='store_true');ap.add_argument('--verify',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.verify:verify();return
    if a.steps%6:raise ValueError('steps must be divisible by six')
    if a.worker:
        r=run(int(a.seeds),a.methods,a.steps,a.batch,a.out)
        print(json.dumps({k:r[k] for k in ('method','seed','train_wall_seconds','updates')}),flush=True);return
    meta={'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),'processor':platform.processor(),'machine':platform.machine(),'steps':a.steps,'batch':a.batch,'seeds':a.seeds,'methods':a.methods,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'thread_environment':{v:os.environ[v] for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS')}}
    (a.out/'environment.json').write_text(json.dumps(meta,indent=2))
    for seed in a.seeds.split(','):
        for method in a.methods.split(','):
            cmd=[sys.executable,__file__,'--worker','--seeds',seed,'--methods',method,'--steps',str(a.steps),'--batch',str(a.batch),'--out',str(a.out)]
            subprocess.run(cmd,check=True)
    rows=[]
    for seed in a.seeds.split(','):
        for method in a.methods.split(','):
            r=json.loads((a.out/f'{method}_{seed}.json').read_text());row={k:v for k,v in r.items() if not isinstance(v,(dict,list)) and k!='memoization'}
            row.update({f'{s}_exact':r['metrics'].get(s,{}).get('exact') for s in ('train','iid','ood','long')});rows.append(row)
    with (a.out/'runs.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('All runs completed:',a.out,flush=True)
if __name__=='__main__':main()
