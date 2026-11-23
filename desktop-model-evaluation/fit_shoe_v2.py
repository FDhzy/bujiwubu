"""Fit and verify a compact logit recalibrator using local catalog photos."""
from pathlib import Path
import json
from hashlib import sha256
import numpy as np
from models.desktop_shoe import classify
d=np.load('shoe-cache.npz')
V,y,f=d['x'],d['y'],d['f']
X=V[:,:3].reshape(-1,15).astype(float)
classes=np.array([0,1,3]); train=f==0; tune=f==1; test=f==2
m=X[train].mean(0); s=np.maximum(X[train].std(0),1e-5); z=(X-m)/s
target=(y[:,None]==classes).astype(float)
a=z[train]; T=target[train]
w=np.zeros((15,3)); b=np.zeros(3)
for _ in range(900):
 q=a@w+b; q=np.exp(q-q.max(1,keepdims=True)); q/=q.sum(1,keepdims=True)
 d=q-T
 w-=.18*(a.T@d/len(a)+.012*w)
 b-=.18*d.mean(0)
q=z@w+b;q=np.exp(q-q.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
forced=classes[q.argmax(1)]; score=q.max(1)
mz=float(np.max([np.quantile(np.max(np.abs(z[train]),axis=1),.99),3.]))
in_range=np.max(np.abs(z),axis=1)<=mz
gates={}
for k in classes:
 for th in [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]:
  accepted=tune & (forced==k) & (score>=th) & in_range
  if accepted.sum()>=12 and np.mean(y[accepted]==k)>=.85:
   gates[str(k)]=th
   break
p=dict(classes=classes.tolist(),mean=m.tolist(),scale=s.tolist(),weights=w.tolist(),bias=b.tolist(),
       thresholds=gates,max_z=mz,base_sha256=sha256(Path('training_checkpoints/shoe_model.tflite').read_bytes()).hexdigest())
pred=[classify(x[:3],p)[0] for x in V[test]]
g=y[test].tolist()
good=[j==k for j,k in zip(pred,g) if j is not None]
metrics=dict(n=len(g),baseline_raw_correct=int((V[test,0].argmax(1)==y[test]).sum()),
             baseline_imagenet_correct=int((V[test,3].argmax(1)==y[test]).sum()),
             forced_correct=int((forced[test]==y[test]).sum()),accepted=len(good),accepted_correct=int(sum(good)),
             per_class={str(i):{'n':sum(k==i for k in g),'accepted_as':sum(j==i for j in pred),
                               'accepted_correct':sum(j==k==i for j,k in zip(pred,g))} for i in classes})
Path('shoe-metrics.json').write_text(json.dumps(metrics,indent=2,default=int)+'\n')
Path('shoe-params.json').write_text(json.dumps(p,indent=2)+'\n')
print(metrics,gates,flush=True)


