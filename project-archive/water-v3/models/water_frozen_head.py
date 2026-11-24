"""Frozen-feature experimental puddle model for the Windows demo."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np

def feature_map(frame, extractor):
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    x=cv2.resize(rgb,(256,256)).astype(np.float32)/255
    local, scene, base=(v.numpy()[0] for v in extractor(x[None],training=False))
    lt=cv2.resize(local,(128,128),interpolation=cv2.INTER_AREA)
    coarse=cv2.resize(scene,(128,128),interpolation=cv2.INTER_LINEAR)
    base=base[:,:,0]
    prob=cv2.resize(base,(128,128),interpolation=cv2.INTER_AREA)
    context=np.r_[scene.mean((0,1)),np.quantile(base,[.25,.5,.75,.9])].astype(np.float32)
    g=np.broadcast_to(context,(128,128,len(context)))
    f=np.concatenate((lt,coarse,prob[:,:,None],g),axis=-1)
    if not np.isfinite(f).all(): raise ValueError("Invalid water activations")
    return f,base

def extractor_for(model_path):
    from tensorflow.keras import Model
    from tensorflow.keras.models import load_model
    m=load_model(str(model_path),compile=False,safe_mode=True)
    return Model(m.inputs,[m.layers[-2].output,m.get_layer('conv2d_11').output,m.output])

def score(f, p):
    coef=np.asarray(p['coef'],dtype=np.float32)
    z=np.einsum('hwk,k->hw',f,coef,optimize=False)+p['bias']
    return 1/(1+np.exp(-np.clip(z,-30,30)))

def mask_for(prob,h,w,t,area):
    z=cv2.resize(prob,(w,h),interpolation=cv2.INTER_LINEAR)
    m=(z>t).astype(np.uint8)
    k=np.ones((7,7),np.uint8)
    m=cv2.morphologyEx(cv2.morphologyEx(m,cv2.MORPH_CLOSE,k),cv2.MORPH_OPEN,k)
    n,l,st,_=cv2.connectedComponentsWithStats(m)
    keep=st[:,cv2.CC_STAT_AREA]>=area*m.size
    keep[0]=False
    return z,keep[l].astype(np.uint8),int(keep.sum())

class FrozenHeadWaterDetector:
    def __init__(self, model_path, params_path):
        self.p=json.loads(Path(params_path).read_text(encoding='utf-8'))
        if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=self.p['backbone_sha256']:
            raise ValueError("Water head and backbone mismatch")
        self.extractor=extractor_for(model_path)

    def detect(self, frame):
        from models.water_detector import WaterDetectionResult
        f,_=feature_map(frame,self.extractor)
        z,m,n=mask_for(score(f,self.p),*frame.shape[:2],self.p['threshold'],self.p['min_area'])
        return WaterDetectionResult(mask=m,confidence=z,coverage_ratio=float(m.mean()),num_regions=n)

