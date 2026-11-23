"""Local calibration/diagnostics. Usage: python eval_water_v2.py ORIGINAL_ROOT."""
from pathlib import Path
import sys, os, time, json
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import cv2, numpy as np
from tensorflow.keras.models import load_model
from models.water_calibration import features, project, image_input, postprocess
root=Path(sys.argv[1])
def im(p,f):return cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),f)
src=root/'training_data/water_real'
pairs=sorted((p,src/'masks'/f'{p.stem}.png') for p in (src/'images').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png') and (src/'masks'/f'{p.stem}.png').exists())
ix=np.random.default_rng(20261123).choice(len(pairs),60,replace=False)
order=np.random.default_rng(100).permutation(60)
obs=root/'training_data/obstacle_yolo/images/val'
orig=sorted(x for x in obs.iterdir() if x.suffix.lower() in ('.jpg','.png','.jpeg'))
ds=[orig[i] for i in np.random.default_rng(719).permutation(len(orig))[:120]]
# Visually screened trial photos: no visible puddle; not an exhaustive audit.
dry_a=[3,11,12,18,21,27,30,32,40,42,62,67,74,80,83,94]
dry_b=[28,46,47,50,65,68,84,95,99,101,102,104,105,108,111,113,114,118]
n1=len(dry_a)
end=60+n1
gt=[]; feats=[]; maps=[]
if not Path('water-ablation-local.npz').exists():
 model=load_model('training_checkpoints/water_model_best.h5',compile=False,safe_mode=True)
 for i,(p,m) in enumerate([pairs[j] for j in ix]+[(ds[j],None) for j in dry_a+dry_b]):
  a=cv2.resize(im(p,1),(640,480)); y=np.zeros((480,640),bool) if m is None else cv2.resize(im(m,0),(640,480),interpolation=cv2.INTER_NEAREST)>127
  rgb=image_input(a); z=model(np.stack((rgb,rgb[:,::-1])),training=False).numpy()[:,:,:,0]; assert np.isfinite(z).all()
  gt.append(y); feats.append(features(rgb,z[0],z[1][:,::-1])); maps.append(z[0])
  if i%10==9: print('cached',i+1,flush=True)
 np.savez_compressed('water-ablation-local.npz',g=gt,f=feats,y=maps)
data=np.load('water-ablation-local.npz')
G,F,Y=data['g'],data['f'],data['y']
tune=list(order[:20])+list(range(60,end))
held=list(order[20:])+list(range(end,len(G)))
r=np.random.default_rng(432)
X=[]; T=[]; W=[]
for i in tune:
 s=r.choice(256**2,6000,False)
 g=cv2.resize(G[i].astype(np.uint8),(256,256),interpolation=cv2.INTER_NEAREST).ravel()[s]
 X.append(F[i].reshape(-1,7)[s]); T.append(g); W.extend([.6 if i>=60 else 1.]*len(g))
x=np.concatenate(X).astype(float); t=np.concatenate(T); w=np.asarray(W)
mu=x.mean(0); sd=np.maximum(x.std(0),1e-5); x=(x-mu)/sd
x=np.column_stack((x,np.ones(len(x))))
p=np.zeros(8)
# small regularized logistic regression, learned only on tuning frames.
weights=w*np.where(t,2.,1.)
for _ in range(12):
 pred=1/(1+np.exp(-np.clip(x@p,-30,30)))
 reg=np.r_[np.full(7,.002),0]
 grad=x.T@(weights*(pred-t))/len(t)+reg*p
 h=(x.T*(weights*pred*(1-pred)))@x/len(t)+np.diag(reg+1e-6)
 p-=np.linalg.solve(h,grad)
coef=p[:-1]/sd; params={'coef':coef.tolist(),'bias':float(p[-1]-mu@coef)}
P=np.array([project(f,params) for f in F])
def metric(maps,th,area,ids):
 I=P0=G0=miss=dry=drypix=0; J=[]
 for i in ids:
  score=cv2.resize(maps[i],(640,480))
  m=postprocess(score,th,area)[0]>0; g=G[i]
  a,b,c=int((m&g).sum()),int(m.sum()),int(g.sum())
  if i>=60: dry+=bool(b);drypix+=b;continue
  I+=a;P0+=b;G0+=c;miss+=not a;J.append(a/max(1,b+c-a))
 return dict(miou=float(np.mean(J)),precision=I/max(1,P0),recall=I/max(1,G0),miss=miss,dry=dry,dry_coverage=drypix/(307200*sum(i>=60 for i in ids)))
base=metric(Y,.6,.01,tune)
print('TUNE BASE',base,flush=True)
variants=[]
for kind,M,thresholds in [('base',Y,[.6,.7,.8,.9,.95]),('flip',F[:,:,:,0],[.6,.7,.8,.9]),('trained',P,[.15,.2,.25,.30,.35,.40,.45,.50,.55,.6])]:
 for th in thresholds:
  for area in (.003,.01):
   s=metric(M,th,area,tune)
   if s['recall']>=base['recall']-.10 and s['miss']<=base['miss']+1:
    variants.append((s['miou']+.18*s['precision']-.25*s['dry_coverage']-.12*s['dry']/n1,kind,th,area,s))
if not variants: raise RuntimeError('No eligible candidate')
_,kind,th,area,s=max(variants,key=lambda v:v[0])
print('CHOSEN',kind,th,area,s,flush=True)
baseline=metric(Y,.6,.01,held)
candidate=metric({'base':Y,'flip':F[:,:,:,0],'trained':P}[kind],th,area,held)
result={'method':kind,'threshold':th,'area':area,'held_positive':40,'held_dry':len(dry_b),'baseline':baseline,'candidate':candidate}
Path('water-metrics.json').write_text(json.dumps(result,indent=2)+'\n')
Path('water-params.json').write_text(json.dumps({**params,'method':kind,'threshold':th,'area':area},indent=2)+'\n')
print('HELD',result,flush=True)

