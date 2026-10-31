# EXPLORATORY post hoc (not in frozen plan). See EXPLORATORIO_categorias_amplias.md
import sys, json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0,'../repo_push/repo/experiments/exp_008_dissociation')
from _shared import CLASS_TO_SUPERCLASS as S
rng=np.random.default_rng(0); B=2000
lg=lambda p: np.log(np.clip(p,.01,.99)/(1-np.clip(p,.01,.99)))
ci=lambda a:[float(np.nanpercentile(a,2.5)),float(np.nanpercentile(a,97.5))]
out={}
# C1 human
h=pd.read_csv('data/h4_long.csv'); h=h[h.in_target==1].copy()
# FIX (DESV §27): unify equivalent condition labels (exps use '3sub' or 'three_subordinate', etc.)
COND={'one':'one','three_subordinate':'sub','3sub':'sub','three_basic':'bas','3bas':'bas','three_superordinate':'sup','3sup':'sup'}
h['condition']=h.condition.map(COND); assert h.condition.notna().all()
def hum_units(df):
    g=df.groupby(['image','condition']).agg(y=('selected','mean'),d=('dist','mean')).reset_index(); return g
g=hum_units(h); sl=lambda g: np.polyfit(g.d,lg(g.y),1)[0]
subs=h.subid.unique(); bys={s:x for s,x in h.groupby('subid')}
bs=[sl(hum_units(pd.concat([bys[s] for s in rng.choice(subs,len(subs))]))) for _ in range(B)]
lev=h.groupby('level').agg(y=('selected','mean'),d=('dist','mean'))
rho=spearmanr(g.d,g.y)[0]
cl=h.groupby(['condition','level']).agg(y=('selected','mean'),d=('dist','mean')).reset_index()
out['C1_human']=dict(slope_logit=float(sl(g)),ci=ci(bs),spearman_unit=float(rho),n_units=len(g),by_level=lev.round(3).to_dict(),by_condition_level={f'{r.condition}|{r.level}':dict(y=round(r.y,3),d=round(r.d,3)) for r in cl.itertuples()})
# C1 learner
t=pd.read_csv('data/tiers_c10.csv'); t['d']=t.d_perc; t['y']=t.acc_tier
idx=[rng.choice(len(t),len(t)) for _ in range(B)]
bl=[np.polyfit(t.d.values[i],lg(t.y.values[i]),1)[0] if len(set(i))>2 else np.nan for i in idx]
tier=t.groupby('tier').agg(y=('y','mean'),d=('d','mean'))
out['C1_learner']=dict(slope_logit=float(np.polyfit(t.d,lg(t.y),1)[0]),ci=ci(bl),spearman_unit=float(spearmanr(t.d,t.y)[0]),n_units=len(t),by_tier=tier.round(3).to_dict(),native_centroid_paper=0.943)
# C2
p=pd.read_csv('data/pairs_c100.csv'); p['same']=[S[a]==S[b] for a,b in zip(p.a,p.b)]
def cd(df,v):
    w=df[df.same][v].dropna(); b=df[~df.same][v].dropna()
    sp=np.sqrt(((len(w)-1)*w.var()+(len(b)-1)*b.var())/(len(w)+len(b)-2)); return (b.mean()-w.mean())/sp,len(w),len(b)
for v in ['d_things','d_hsj','d_perc','acc_pair']:
    dd=p.dropna(subset=[v]); est,nw,nb=cd(dd,v); W=dd[dd.same]; Bt=dd[~dd.same]
    bb=[cd(pd.concat([W.sample(len(W),replace=True,random_state=k),Bt.sample(len(Bt),replace=True,random_state=k+B)]),v)[0] for k in range(B)]
    out[f'C2_{v}']=dict(cohen_d_between_minus_within=float(est),ci=ci(bb),n_within=nw,n_between=nb,mean_within=float(W[v].mean()),mean_between=float(Bt[v].mean()))
# C3
w=p[p.same].copy(); w['sc']=[S[a] for a in w.a]
prof=w.groupby('sc')[['d_perc','d_things','d_hsj']].mean()
for v in ['d_things','d_hsj']:
    q=prof[['d_perc',v]].dropna(); r=spearmanr(q.d_perc,q[v])[0]
    bb=[spearmanr(*q.iloc[rng.choice(len(q),len(q))].T.values)[0] for _ in range(B)]
    out[f'C3_{v}']=dict(spearman=float(r),ci=ci(bb),n_superclasses=len(q))
json.dump(out,open('out/exploratorio_categorias_amplias.json','w'),indent=1)
print(json.dumps(out,indent=1))
# Figures
plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
f,ax=plt.subplots(1,2,figsize=(7.6,3.1))
CN={'one':'1 sub example','sub':'3 sub examples','bas':'3 basic examples','sup':'3 superord. examples'}; MK={'sub':'o','bas':'s','sup':'^'}
ax[0].scatter(g.d,g.y,s=8,c='0.75',label='image × condition (n=%d)'%len(g))
for c,col in zip(['one','sub','bas','sup'],['#1b9e77','#d95f02','#7570b3','#e7298a']):
    q=cl[cl.condition==c].set_index('level').loc[['sub','bas','sup']]; ax[0].plot(q.d,q.y,'-',c=col,lw=1.2,label=CN[c])
    for l,r in q.iterrows(): ax[0].plot(r.d,r.y,MK[l],c=col,ms=4)
ax[0].text(0.02,0.98,'markers: ○ sub  □ basic  △ superord. test item',transform=ax[0].transAxes,fontsize=6,va='top')
ax[0].set(title='(a) Humans (Lewis & Frank 2018), target items',xlabel='DINOv2 distance to training exemplars',ylabel='Proportion of test items selected',ylim=(-0.02,1.02))
ax[1].scatter(t.d,t.y,s=14,c='0.6',label='disjunctive word')
tt=tier.loc[['near','mid','far']]; ax[1].plot(tt.d,tt.y,'o-',c='k',label='tier mean')
for l,r in tt.iterrows(): ax[1].annotate(l,(r.d,r.y),xytext=(4,4),textcoords='offset points')
ax[1].axhline(1/3,ls=':',c='0.5'); ax[1].text(0.72,0.35,'chance',fontsize=7,color='0.4')
ax[1].set(title='(b) Learner (centroid), CIFAR-10 tiers',xlabel='DINOv2 distance between member categories ($d_{member}$)',ylabel='Naming accuracy (3-way)')
ax[0].set_xlim(0.1,1.0); ax[1].set_xlim(0.2,1.0)
for a in ax: a.legend(frameon=False,fontsize=6,loc='center left')
f.text(0.99,-0.03,'EXPLORATORY (post hoc). Different y quantities: (a) extension of a word (selection), (b) naming accuracy; distances also differ.',ha='right',fontsize=7,style='italic')
f.tight_layout(); [f.savefig(f'figs/comparativos/exploratorio/x1_breadth_gradient_humans_vs_learner.{e}',dpi=300,bbox_inches='tight') for e in ['png','pdf']]
f,ax=plt.subplots(1,4,figsize=(8,2.6))
for a,v,lab in zip(ax,['d_things','d_hsj','d_perc','acc_pair'],['Human dist. THINGS','Human dist. HSJ','DINOv2 distance','Learner accuracy (exp. 008)']):
    dd=p.dropna(subset=[v]); a.boxplot([dd[dd.same][v],dd[~dd.same][v]],widths=.5,showfliers=False)
    a.set_xticks([1,2],[f"within\n(n={dd.same.sum()})",f"between\n(n={(~dd.same).sum()})"]); a.set_title(lab,fontsize=8)
    o=out[f'C2_{v}']; a.text(.5,.02,f"d={o['cohen_d_between_minus_within']:.2f} [{o['ci'][0]:.2f},{o['ci'][1]:.2f}]",transform=a.transAxes,ha='center',fontsize=6.5)
f.suptitle('CIFAR-100 superclasses: within vs between',fontsize=9)
f.text(0.99,-0.04,'EXPLORATORY (post hoc)',ha='right',fontsize=7,style='italic')
f.tight_layout(); [f.savefig(f'figs/comparativos/exploratorio/x2_superclass_within_between.{e}',dpi=300,bbox_inches='tight') for e in ['png','pdf']]
f,ax=plt.subplots(1,2,figsize=(6,2.8))
for a,v,lab in zip(ax,['d_things','d_hsj'],['THINGS','HSJ']):
    q=prof[['d_perc',v]].dropna(); a.scatter(q.d_perc,q[v],s=12,c='k')
    for n,r in q.iterrows(): a.annotate(n.replace('_',' ')[:14],(r.d_perc,r[v]),fontsize=4.5,color='0.4')
    o=out[f'C3_{v}']; a.set(xlabel='Mean within-superclass DINOv2 distance',ylabel=f'Mean within-superclass human dist. ({lab})',title=f"ρ={o['spearman']:.2f} [{o['ci'][0]:.2f},{o['ci'][1]:.2f}], n={o['n_superclasses']}")
f.text(0.99,-0.04,'EXPLORATORY (post hoc)',ha='right',fontsize=7,style='italic')
f.tight_layout(); [f.savefig(f'figs/comparativos/exploratorio/x3_superclass_profile.{e}',dpi=300,bbox_inches='tight') for e in ['png','pdf']]
