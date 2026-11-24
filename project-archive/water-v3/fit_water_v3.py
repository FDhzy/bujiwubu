"""Fit only the water segmentation head; no backbone training or data uploads."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import sys, json, hashlib
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from models.water_frozen_head import feature_map,extractor_for,score,mask_for
src=Path(sys.argv[1])/'training_data'
model_path=Path(sys.argv[2])/'training_checkpoints/water_model_best.h5'
parent=Path(sys.argv[2])
def im(p,f):return cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),f)
pairs=sorted((p,src/'water_real/masks'/f'{p.stem}.png') for p in (src/'water_real/images').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png') and (src/'water_real/masks'/f'{p.stem}.png').exists())
old=np.random.default_rng(20261123).choice(len(pairs),60,replace=False)
rem=np.setdiff1d(np.arange(len(pairs)),old)
order=np.random.default_rng(100).permutation(60)
rng=np.random.default_rng(24009)
new=rng.permutation(rem)
dry=[Path(a) for a in (parent/'water_dry_candidates_local.txt').read_text(encoding='utf-8').splitlines()]
da=[3,11,12,18,21,27,30,32,40,42,62,67,74,80,83,94]
db=[28,46,47,50,65,68,84,95,99,101,102,104,105,108,111,113,114,118]
# The 18 same dry challenge images are never included in fitting or threshold selection.
training=[pairs[i] for i in new[:110]]+[(dry[i],None) for i in da[:12]]
tuning=[pairs[old[i]] for i in order[:20]]+[(dry[i],None) for i in da[12:]]
held=[pairs[old[i]] for i in order[20:]]+[(dry[i],None) for i in db]
fresh=[pairs[i] for i in new[110:140]]
net=extractor_for(model_path)
def load(p,m):
    a=cv2.resize(im(p,1),(640,480))
    y=np.zeros((480,640),bool) if m is None else cv2.resize(im(m,0),(640,480),interpolation=cv2.INTER_NEAREST)>127
    f,b=feature_map(a,net)
    return f,b,y
X=[];Y=[];W=[]
for i,(p,m) in enumerate(training):
    f,b,g=load(p,m)
    gg=cv2.resize(g.astype('uint8'),(128,128),interpolation=cv2.INTER_AREA).ravel()>0
    for val in (False,True):
        j=np.flatnonzero(gg==val)
        if not len(j):continue
        jj=rng.choice(j,min(len(j),480),replace=False)
        X.append(f.reshape(-1,f.shape[-1])[jj]);Y.append(gg[jj])
        W.extend([((2.0 if val else 1.0) if m is not None else 3.0)*480/len(jj)]*len(jj))
    if (i+1)%20==0:print('fit frames',i+1,flush=True)
x=np.concatenate(X);y=np.concatenate(Y).astype('float32');w=np.asarray(W,np.float32);w/=w.sum()
mu=x.mean(0);std=np.maximum(x.std(0),.01)
x=np.ascontiguousarray((x-mu)/std)
def fit(reg):
    def loss(p):
        p=p.astype('float32')
        z=x@p[:-1]+p[-1]
        e=w*(expit(z)-y)
        return float(w@(np.logaddexp(0,z)-y*z)+.5*reg*(p[:-1]@p[:-1])),np.r_[x.T@e+reg*p[:-1],e.sum()].astype(float)
    opt=minimize(loss,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':5e-7})
    p=opt.x
    coef=p[:-1]/std
    return dict(coef=coef.tolist(),bias=float(p[-1]-mu@coef))
heads=[fit(x) for x in (.012,.04)]
def data(items,pp):
    for p,m in items:
        f,b,g=load(p,m)
        yield ([b]+[score(f,c) for c in pp],g,m is None)
def metrics(rows,j,t,area):
    I=P=G=miss=fp=nds=pix=0;J=[]
    for ps,g,dry in rows:
        mask=mask_for(ps[j],*g.shape,t,area)[1].astype(bool)
        if dry:
            fp+=bool(mask.any());nds+=1;pix+=int(mask.sum());continue
        a,b,c=map(int,((mask&g).sum(),mask.sum(),g.sum()))
        I+=a;P+=b;G+=c;miss+=a==0;J.append(a/max(b+c-a,1))
    return dict(n_positive=len(J),n_dry=nds,miou=float(np.mean(J)),precision=I/max(P,1),recall=I/max(G,1),miss=miss,
                dry_fp=fp,dry_area=pix/(max(nds,1)*640*480))
tune=list(data(tuning,heads))
base=metrics(tune,0,.60,.01); print('tune baseline',base,flush=True)
opts=[]
for j in (1,2):
    for t in (.3,.4,.5,.55,.6,.65,.7,.75,.8,.85):
        for a in (.005,.01):
            s=metrics(tune,j,t,a)
            if s['recall']>=base['recall']-.04 and s['miss']<=base['miss']:
                opts.append((s['miou']+.15*s['precision']-.30*s['dry_area']-.07*s['dry_fp'],j,t,a,s))
if not opts:
    print('NO_CANDIDATE_MET_TUNING_RECALL',flush=True)
    raise SystemExit(2)
_,j,t,a,m=max(opts,key=lambda row:row[0])
p=heads[j-1];p.update(threshold=t,min_area=a,backbone_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())
report={'chosen_on_tuning_only':m,'threshold':t,'min_area':a}
for label,items in (('matched_holdout',held),('extra_positive_holdout',fresh)):
    r=list(data(items,[p]))
    report[label]={'baseline':metrics(r,0,.60,.01),'candidate':metrics(r,1,t,a)}
    print(label,report[label],flush=True)
Path('water-v3-head.json').write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8')
Path('water-v3-results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

