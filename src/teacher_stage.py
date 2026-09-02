import gc, json, math, os, random, shutil, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from config import *
from common import atomic_json, release_cuda, seed_all

def load_teacher(source,dtype):
    m=AutoModelForCausalLM.from_pretrained(str(source),torch_dtype=dtype,low_cpu_mem_usage=True); m.config.use_cache=False; return m

def ckpt_root(p,t): return p['teachers']/f'teacher_{t}'/'checkpoints'
def find_ckpt(p,t):
    root=ckpt_root(p,t); pointer=root/'latest.json'
    if not pointer.exists(): return None
    try:
        info=json.loads(pointer.read_text()); c=root/info['checkpoint']
        return c if (c/'model').exists() and (c/'state.pt').exists() else None
    except Exception: return None

def save_ckpt(p,t,model,opt,sched,scaler,step,pos,losses):
    torch.cuda.synchronize(); root=ckpt_root(p,t); root.mkdir(parents=True,exist_ok=True); name=f'step_{step:06d}'; final=root/name; temp=root/f'.tmp_{name}_{os.getpid()}'
    shutil.rmtree(temp,ignore_errors=True); shutil.rmtree(final,ignore_errors=True); temp.mkdir()
    model.save_pretrained(str(temp/'model'),safe_serialization=True)
    torch.save({'step':step,'next_position':pos,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'scaler':scaler.state_dict(),'losses':losses,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temp/'state.pt')
    temp.rename(final); pointer=root/'latest.json'; previous=None
    if pointer.exists():
        try: previous=json.loads(pointer.read_text()).get('checkpoint')
        except Exception: pass
    atomic_json(pointer,{'checkpoint':name,'step':step})
    if previous and previous!=name: shutil.rmtree(root/previous,ignore_errors=True)
    print(f'[teacher {t}] checkpoint @ {step}')

def train_teachers(p):
    print('\n'+'='*80+'\nSTAGE 2: FULL TEACHER FINE-TUNING\n'+'='*80); summary=[]
    for t in range(NUM_TEACHERS):
        run=p['teachers']/f'teacher_{t}'; final=run/'final_model'; done=run/'DONE.json'; run.mkdir(parents=True,exist_ok=True)
        if final.exists() and done.exists(): summary.append(json.loads(done.read_text())); print(f'[teacher {t}] COMPLETE; skip'); continue
        ds=load_from_disk(str(p['data']/f'teacher_{t}')); counts=np.bincount(np.asarray(ds['domain_id']),minlength=NUM_DOMAINS)
        if not np.all(counts==TEACHER_BLOCKS_PER_DOMAIN): raise RuntimeError(f'T{t}: data not balanced')
        n=len(ds); total=math.ceil(n/TEACHER_BATCH_SIZE); warm=max(1,round(total*TEACHER_WARMUP_FRACTION)); c=find_ckpt(p,t); seed_all(SEED+200000+t)
        if c is None:
            model=load_teacher(TEACHER_MODEL,torch.float32).to('cuda'); step=0; pos=0; losses=[]; state=None; print(f'[teacher {t}] START {n:,} blocks / {n*BLOCK_SIZE:,} tokens')
        else:
            state=torch.load(c/'state.pt',map_location='cpu',weights_only=False); model=load_teacher(c/'model',torch.float32).to('cuda'); step=int(state['step']); pos=int(state['next_position']); losses=list(state['losses']); print(f'[teacher {t}] RESUME step={step}/{total}')
        model.train(); opt=torch.optim.AdamW(model.parameters(),lr=TEACHER_LR,weight_decay=.01,foreach=False); sched=get_cosine_schedule_with_warmup(opt,num_warmup_steps=warm,num_training_steps=total); scaler=torch.amp.GradScaler('cuda')
        if state is not None:
            opt.load_state_dict(state['optimizer'])
            for s in opt.state.values():
                for k,v in list(s.items()):
                    if torch.is_tensor(v): s[k]=v.to('cuda')
            sched.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler']); random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        started=time.time()
        while pos<n:
            rows=ds[pos:min(pos+TEACHER_BATCH_SIZE,n)]; ids=torch.tensor(rows['input_ids'],dtype=torch.long).pin_memory().to('cuda',non_blocking=True); mask=torch.ones_like(ids); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16): out=model(input_ids=ids,attention_mask=mask,labels=ids,use_cache=False); loss=out.loss
            if not torch.isfinite(loss): raise RuntimeError(f'T{t}: non-finite loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); scaler.step(opt); scaler.update(); sched.step(); torch.cuda.synchronize()
            pos+=ids.shape[0]; step+=1; losses.append(float(loss.detach().cpu()))
            if step==1 or step%10==0: print(f'[teacher {t}] {step:4d}/{total} examples={pos:5d}/{n} loss={np.mean(losses[-10:]):.4f} gpu={torch.cuda.memory_allocated()/1024**3:.2f}GiB')
            del rows,ids,mask,out,loss
            if step%TEACHER_CHECKPOINT_EVERY==0: save_ckpt(p,t,model,opt,sched,scaler,step,pos,losses)
        torch.cuda.synchronize(); shutil.rmtree(final,ignore_errors=True); model.half(); model.save_pretrained(str(final),safe_serialization=True)
        info={'teacher':t,'complete':True,'blocks':n,'tokens':n*BLOCK_SIZE,'domain_counts':counts.tolist(),'optimizer_steps':step,'final_recent_loss':float(np.mean(losses[-20:])),'runtime_minutes':(time.time()-started)/60}; atomic_json(done,info); summary.append(info); shutil.rmtree(ckpt_root(p,t),ignore_errors=True); print(f'[teacher {t}] FULLY TRAINED')
        del model,opt,sched,scaler,ds; release_cuda()
    pd.DataFrame(summary).to_csv(p['root']/'teacher_training.csv',index=False); print('[teachers] all COMPLETE')
