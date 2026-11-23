from pathlib import Path
import os, re, sys, json, cv2, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
import tensorflow as tf
from hashlib import sha256
root=Path(sys.argv[1]); out=Path(sys.argv[2])
def read(p): return cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),1)
def cache_shoes():
 d=root/'training_data/shoe_real'
 net=tf.lite.Interpreter(model_content=(out/'training_checkpoints/shoe_model.tflite').read_bytes(),num_threads=4)
 net.allocate_tensors(); i=net.get_input_details()[0]['index']; o=net.get_output_details()[0]['index']
 xs=[]; ys=[]; folds=[]; seen=set()
 # Hold contiguous index blocks out, reducing adjacency leakage. Exact copies cannot cross folds.
 for cl in (0,1,3):
  files=[]
  for p in sorted((d/'images').glob('*.jpg')):
   lab=d/'labels'/(p.stem+'.txt')
   if lab.exists() and lab.read_text().strip()==str(cl): files.append(p)
  for fold,(lo,hi,count) in enumerate([(0,850,75),(900,1250,40),(1450,2000,60)]):
   for p in np.random.default_rng(875+cl*7+fold).permutation(files[lo:hi]):
    data=p.read_bytes(); h=sha256(data).digest()
    if h in seen: continue
    seen.add(h)
    im=cv2.imdecode(np.frombuffer(data,np.uint8),1)
    if im is None: continue
    rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    full=cv2.resize(rgb,(224,224)).astype(np.float32)/255
    pad=cv2.resize(rgb,(256,256))[16:240,16:240].astype(np.float32)/255
    flip=np.ascontiguousarray(full[:,::-1])
    norm=(full-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    feat=[]
    for a in (full,pad,flip,norm):
     net.set_tensor(i,a[None]);net.invoke();feat.append(net.get_tensor(o)[0])
    xs.append(feat);ys.append(cl);folds.append(fold)
    if sum(a==cl and b==fold for a,b in zip(ys,folds))>=count:break
   print('shoe',cl,fold,len(xs),flush=True)
 np.savez_compressed(out/'shoe-cache.npz',x=xs,y=ys,f=folds)
cache_shoes()

