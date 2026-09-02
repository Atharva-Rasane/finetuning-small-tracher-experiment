import gc, json, os, random
from pathlib import Path
import numpy as np
import torch
from config import *

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def atomic_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(data,f,indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def atomic_torch(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+'.tmp')
    torch.save(obj,tmp); os.replace(tmp,path)

def atomic_numpy(arr,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:
        np.save(f,arr); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def release_cuda():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def paths(root):
    root=Path(root).resolve()
    p={
        'root':root,'data':root/'datasets','parts':root/'datasets'/'parts','teachers':root/'teachers',
        'scores':root/'scores','soft':root/'softlabels','students':root/'students','reports':root/'reports',
        'student_data':root/'datasets'/'student_train','val_data':root/'datasets'/'validation',
        'prepared':root/'datasets'/'PREPARED.json','soft_file':root/'softlabels'/'oracle_softlabels_top32.pt',
        'initial_adapter':root/'students'/'common_initial_adapter','finished':root/'EXPERIMENT_FINISHED.json'
    }
    for k in ('root','data','parts','teachers','scores','soft','students','reports'): p[k].mkdir(parents=True,exist_ok=True)
    return p
