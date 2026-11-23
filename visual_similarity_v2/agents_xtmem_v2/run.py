"""Run with python run.py PATH_TO_REPO --out outputs. See protocol.md."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

SEED = 42
TOL = 1e-6
COMMIT = '23e4082b0bd7a33889714d0d179fa5a12791fb6c'
ENCODERS = ['R0_dinov2s','O1_dinov2b','R1_cvcl','E1_siglip','O2_openclip','E2_vitsup','E3_rn50sup','E4_mae','F1_rn50rand0','F2_pixels',
            'F1_rn50rand1','F1_rn50rand2','E4s_mae_cls']
DOMAINS = {'vegetables':'c1','vehicles':'c2','animals':'c3'}
CONDITIONS = {'one':['sub3'],'3sub':['sub3','sub4','sub5'],'3bas':['sub3','bas3','bas4'],'3sup':['sub3','sup5','sup6']}
CANONICAL = {'three_subordinate':'3sub','three_basic':'3bas','three_superordinate':'3sup'}
TEST = [f'{prefix}_{level}{i}' for prefix in DOMAINS.values() for level,n in [('sub',2),('bas',2),('sup',4)] for i in range(1,n+1)]
PAIRS = [
    ('vehicles','c2_sup4','c3_sup2','gray car / turtle'),
    ('animals','c3_sup1','c1_sup3','rabbit / radish'),
    ('animals','c3_sup2','c3_sup1','turtle / rabbit'),
    ('vegetables','c1_sup3','c1_sup4','radish / broccoli'),
    ('vehicles','c2_sup3','c2_sup4','blue car / gray car')
]

def geometry(path):
    with np.load(path) as z:
        a=z['emb'].astype(np.float64)
        a/=np.linalg.norm(a,axis=1,keepdims=True)
        assert np.isfinite(a).all() and len(set(z['names']))==45
        return dict(zip(z['names'],a))

def class_label(index):
    return list(DOMAINS)[index]

def read_humans(folder, enc):
    df=pd.read_csv(folder/f'h4_long_{enc}.csv')
    assert len(df)==172800 and df.subid.nunique()==520
    df['condition']=df.condition.replace(CANONICAL)
    return df

def verify_input(repo, directory, relative_path):
    """Check the working file against the exact Git blob used in the original run."""
    expected = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', f'{COMMIT}:{relative_path}'], text=True
    ).strip()
    actual = subprocess.check_output(
        ['git', 'hash-object', '--no-filters', str(directory / relative_path)], text=True
    ).strip()
    if actual != expected:
        raise ValueError(f'{relative_path} differs from the input at {COMMIT}')

def ranked_pair(s1,s2):
    if s1==s2 or abs(s1-s2)<=TOL:
        return 'tie'
    return 'A' if s1>s2 else 'B'

def make_tables(rows,pred,base):
    scores=pd.DataFrame(rows)
    names=pd.DataFrame(pred)
    scores.to_csv(base/'one_word_items.csv',index=False)
    names.to_csv(base/'three_word_items.csv',index=False)
    pairrows=[]
    ix=['encoder','learner','task','domain','condition']
    common=['encoder','learner','domain','condition','image','human_n','human_yes']
    one=scores[common+['score','accepted']].assign(task='one_word')
    three=names[common+['target_score','predicts_target']].rename(columns={'target_score':'score','predicts_target':'accepted'}).assign(task='three_words')
    merged=pd.concat([one,three],ignore_index=True)
    for key,g in merged[merged.condition=='3sup'].groupby(ix):
        h=g.set_index('image')
        for dom,a,b,label in PAIRS:
            if dom!=key[3]: continue
            u,v=h.loc[a],h.loc[b]
            pairrows.append(dict(zip(ix,key),pair=label,image_A=a,image_B=b,score_A=u.score,score_B=v.score,higher_score=ranked_pair(u.score,v.score),accept_A=u.accepted,accept_B=v.accepted,human_A=u.human_yes,human_B=v.human_yes,n_sessions=u.human_n))
    pairs=pd.DataFrame(pairrows)
    pairs.to_csv(base/'broad_teaching_pairs.csv',index=False)

    g=merged[merged.condition.isin(['3sub','3sup'])].copy()
    g['candidate_set']=np.where(g.image.str[:2]!=g.domain.map(DOMAINS),'outside_domain',np.where(g.image.str[3:6]=='sup','target_superordinate','other_target'))
    g.groupby(['encoder','learner','task','domain','candidate_set','condition'],as_index=False).agg(items=('accepted','size'),items_with_decision=('accepted','count'),model_fraction=('accepted','mean'),human_fraction=('human_yes',lambda v:v.sum()/(600*len(v)))).to_csv(base/'condition_by_domain.csv',index=False)
    p=g[g.candidate_set=='target_superordinate']
    effect=p.groupby(['encoder','learner','task','condition']).agg(model=('accepted','mean'),human=('human_yes',lambda v:v.sum()/(600*len(v))),image_conditions=('image','size')).unstack('condition')
    effect.columns=['_'.join(v) for v in effect.columns]
    effect['model_broad_minus_narrow']=effect.model_3sup-effect.model_3sub
    effect['human_broad_minus_narrow']=effect.human_3sup-effect.human_3sub
    effect=effect.reset_index()
    bounds=[]
    for (enc,lr),s in p[(p.task=='one_word')&(p.learner!='random')].groupby(['encoder','learner']):
        wide=s[s.condition=='3sup'].score.to_numpy()
        narrow=s[s.condition=='3sub'].score.to_numpy()
        assert len(wide)==len(narrow)==12
        th=np.r_[-np.inf,np.unique(np.r_[wide,narrow]),np.inf]
        increments=(wide[:,None]>=th).mean(axis=0)-(narrow[:,None]>=th).mean(axis=0)
        bounds.append({'encoder':enc,'learner':lr,'task':'one_word','max_jump_any_fixed_threshold':increments.max()})
    effect.merge(pd.DataFrame(bounds),how='left',on=['encoder','learner','task']).to_csv(base/'broad_minus_narrow.csv',index=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('repo',type=Path);p.add_argument('--out',type=Path,default=Path('outputs'))
    p.add_argument('--learners-root',type=Path,help='Optional source mirror with exactly the same git blobs')
    args=p.parse_args();repo=args.repo.resolve();out=args.out.resolve();out.mkdir(exist_ok=True,parents=True)
    src=(args.learners_root or repo).resolve()
    for mod in ['__init__','base','centroid','kcentroid','exemplar_knn','logreg','linsvm','random_baseline']:
        verify_input(repo, src, f'src/learners/{mod}.py')
    sys.path.insert(0,str(src))
    from src.learners.centroid import CentroidLearner
    from src.learners.kcentroid import KCentroidLearner
    from src.learners.exemplar_knn import ExemplarKNN
    from src.learners.logreg import LogRegLearner
    from src.learners.linsvm import LinSVMLearner
    from src.learners.random_baseline import RandomLearner

    learners={'centroid':(CentroidLearner,{}),'kcentroid2':(KCentroidLearner,{'k':2}),'knn3':(ExemplarKNN,{'k':3,'weighted':True}),'logreg':(LogRegLearner,{'C':1}),'linsvm':(LinSVMLearner,{'C':1}),'random':(RandomLearner,{})}
    single=['centroid','kcentroid2','knn3','random']
    def train(lr,support,labels,k):
        cls,kw=learners[lr]
        model=cls(k,seed=SEED,**kw); model.fit(support,np.array(labels)); return model
    folder=repo/'encoder_equivalence_v2/encoder_equivalence'
    for enc in ENCODERS:
        verify_input(repo, repo, (folder/'out'/f'emb_{enc}.npz').relative_to(repo))
        verify_input(repo, repo, (folder/'data'/f'h4_long_{enc}.csv').relative_to(repo))
    ref=read_humans(folder/'data',ENCODERS[0])
    refrows=ref.drop(columns='dist')
    human=ref.groupby(['category','condition','image']).selected.agg(['sum','count'])
    assert len(human)==288 and (human['count']==600).all()
    rows=[];pred=[];checks={}; hashes={}
    for enc in ENCODERS:
        npz=folder/'out'/f'emb_{enc}.npz'; hashes[enc]=hashlib.sha256(npz.read_bytes()).hexdigest()
        vec=geometry(npz);test=np.stack([vec[n] for n in TEST]);err=0.
        raw=ref if enc==ENCODERS[0] else read_humans(folder/'data',enc)
        assert raw.drop(columns='dist').equals(refrows)
        spans=raw.groupby(['category','condition','image']).dist.agg(['min','max'])
        for cond,ids in CONDITIONS.items():
            supports={d:np.stack([vec[f'{prefix}_{x}'] for x in ids]) for d,prefix in DOMAINS.items()}
            for dom,support in supports.items():
                labels=np.zeros(len(ids),dtype=int)
                for lr in single:
                    model=train(lr,support,labels,1)
                    scores=model.score(test)[:,0]
                    accepted=[None]*len(TEST)
                    for i,x in enumerate(test):
                        if len(ids)==3 and lr!='random':
                            four=np.vstack([support,x])
                            held=[train(lr,np.delete(four,j,axis=0),labels,1).score(four[j])[0] for j in range(3)]
                            accepted[i]=int(scores[i]>=min(held)-TOL)
                        raw_h=human.loc[(dom,cond,TEST[i])]
                        rows.append(dict(encoder=enc,learner=lr,domain=dom,condition=cond,image=TEST[i],score=scores[i],accepted=accepted[i],human_n=raw_h['count'],human_yes=raw_h['sum']))
                    if lr=='centroid':
                        z=spans.loc[(dom,cond)].loc[TEST]
                        err=max(err,float(np.abs(1-scores-z['min']).max()),float(np.abs(1-scores-z['max']).max()))
            support_all=np.vstack(list(supports.values()))
            labs=np.repeat(np.arange(3),len(ids))
            for lr in learners:
                model=train(lr,support_all,labs,3)
                all_scores=model.score(test);winner=model.predict(test)
                assert np.array_equal(winner,all_scores.argmax(axis=1))
                for i,img in enumerate(TEST):
                    for di,dom in enumerate(DOMAINS):
                        raw_h=human.loc[(dom,cond,img)]
                        pred.append(dict(encoder=enc,learner=lr,domain=dom,condition=cond,image=img,target_score=all_scores[i,di],predicted_domain=class_label(winner[i]),predicts_target=int(winner[i]==di),human_n=raw_h['count'],human_yes=raw_h['sum']))
        assert err<=1e-5,(enc,err)
        checks[enc]=err
    assert len(rows)==13*4*12*24 and len(pred)==13*6*12*24
    make_tables(rows,pred,out)
    meta=dict(commit=COMMIT,xtmem='220ed17ec2b85d30a4053711209252fdbc3cb166',seed=SEED,primary_encoders=ENCODERS[:10],other_variants=ENCODERS[10:],python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,centroid_max_error_vs_frozen=checks,embedding_sha256=hashes,one_word_scores=len(rows),three_word_target_rows=len(pred),three_word_classifications=len(pred)//3)
    (out/'run.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Scores:',len(rows),'three-class decisions:',len(pred)//3,'max centroid discrepancy:',max(checks.values()))

if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
