"""Desktop photo suggestions only. Never sets a shoe type or safety threshold."""
import json
from pathlib import Path
import cv2
import numpy as np

SHOE_LABELS = {0:"凉鞋／拖鞋", 1:"低帮鞋", 3:"靴类（请手选具体筒高）"}

def as_features(pred):
    a=np.asarray(pred,dtype=np.float64)
    if a.shape!=(3,5) or not np.isfinite(a).all():
        raise ValueError("Invalid shoe output")
    return a.ravel()

def classify(pred, p):
    x=(as_features(pred)-p['mean'])/p['scale']
    z=x@np.asarray(p['weights'])+p['bias']
    q=np.exp(z-z.max()); q/=q.sum()
    j=int(q.argmax()); cl=p['classes'][j]
    score=float(q[j])
    threshold=p['thresholds'].get(str(cl))
    if threshold is None or score<threshold or np.max(np.abs(x))>p['max_z']:
        return None, score
    return cl, score

def get_views(bgr):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    full=cv2.resize(rgb,(224,224)).astype(np.float32)/255
    crop=cv2.resize(rgb,(256,256))[16:240,16:240].astype(np.float32)/255
    return full,crop,np.ascontiguousarray(full[:,::-1])

class DesktopShoeSuggestion:
    def __init__(self, model_path, params_path):
        import tensorflow as tf
        from hashlib import sha256
        self.params=json.loads(Path(params_path).read_text(encoding='utf-8'))
        buf=Path(model_path).read_bytes()
        if sha256(buf).hexdigest()!=self.params['base_sha256']:
            raise ValueError("Shoe calibration is for a different checkpoint")
        self.net=tf.lite.Interpreter(model_content=buf,num_threads=3)
        self.net.allocate_tensors()
        i,o=self.net.get_input_details()[0],self.net.get_output_details()[0]
        if tuple(i['shape'])!=(1,224,224,3) or tuple(o['shape'])!=(1,5):
            raise ValueError("Unexpected shoe model dimensions")
        self.i,self.o=i['index'],o['index']

    def suggest(self,bgr):
        if bgr is None or bgr.ndim!=3 or min(bgr.shape[:2])<40:
            return None, 0.
        preds=[]
        for view in get_views(bgr):
            self.net.set_tensor(self.i,view[None])
            self.net.invoke()
            preds.append(self.net.get_tensor(self.o)[0])
        return classify(preds,self.params)

