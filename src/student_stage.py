import json, os, random, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from config import *
from common import atomic_json, release_cuda, seed_all
from score_stage import collate

def create_common_adapter(p):
    adapter=p['initial_adapter']
    if adapter.exists() and (adapter/'adapter_config.json').exists() and any(adapter.glob('adapter_model.*')):
        print('[student-init] cached'); return
    if adapter.exists(): shutil.rmtree(adapter,ignore_errors=True)
    print('\n'+'='*80+'\nSTAGE 5: COMMON STUDENT INITIALIZATION\n'+'='*80); seed_all(12345)
    base=AutoModelForCausalLM.from_pretrained(STUDENT_MODEL,torch_dtype=torch.float16,low_cpu_mem_usage=True); base.config.use_cache=False
    cfg=LoraConfig(r=STUDENT_LORA_R,lora_alpha=STUDENT_LORA_ALPHA,lora_dropout=0.0,bias='none',task_type='CAUSAL_LM',target_modules=['c_attn','c_proj'],fan_in_fan_out=True)
    model=get_peft_model(base,cfg); temp=adapter.parent/f'.tmp_{adapter.name}_{os.getpid()}'; shutil.rmtree(temp,ignore_errors=True); model.save_pretrained(str(temp),safe_serialization=True); temp.rename(adapter); del model,base; release_cuda(); print('[student-init] COMPLETE')

def load_student(adapter):
    base=AutoModelForCausalLM.from_pretrained(STUDENT_MODEL,torch_dtype=torch.float16,low_cpu_mem_usage=True); base.config.use_cache=False; base.gradient_checkpointing_enable(); base.enable_input_require_grads(); return PeftModel.from_pretrained(base,str(adapter),is_trainable=True).to('cuda')

@torch.inference_mode()
def evaluate(model,val):
    model.eval(); loader=DataLoader(val,batch_size=2,shuffle=False,collate_fn=collate,num_workers=2,pin_memory=True); total=0.; n=0
    for batch in loader:
        ids=batch['input_ids'].to('cuda',non_blocking=True); mask=batch['attention_mask'].to('cuda',non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16): out=model(input_ids=ids,attention_mask=mask,labels=ids,use_cache=False)
        total+=float(out.loss.detach().cpu())*ids.shape[0]; n+=ids.shape[0]; del ids,mask,out
    model.train(); del loader; release_cuda(); return total/n

def kd_loss(store,student_logits,eid):
    ti=store['top_indices'][eid].to('cuda',dtype=torch.long,non_blocking=True).unsqueeze(0); tl=store['top_logits'][eid].to('cuda',dtype=torch.float32,non_blocking=True).unsqueeze(0); tz=store['log_z_T'][eid].to('cuda',dtype=torch.float32,non_blocking=True).unsqueeze(0); s=student_logits[:,:-1,:].float()
    tlog=tl/TEMPERATURE-tz.unsqueeze(-1); tp=torch.exp(tlog); ttail=(1-tp.sum(-1)).clamp(1e-7,1.0); slogz=torch.logsumexp(s/TEMPERATURE,dim=-1); stop=s.gather(-1,ti); slog=stop/TEMPERATURE-slogz.unsqueeze(-1); sp=torch.exp(slog); stail=(1-sp.sum(-1)).clamp(1e-7,1.0); kl=(tp*(tlog-slog)).sum(-1)+ttail*(torch.log(ttail)-torch.log(stail)); return kl.mean()*TEMPERATURE**2

def ckroot(p,mode): return p['students']/mode/'checkpoints'
def find_ckpt(p,mode):
    root=ckroot(p,mode); ptr=root/'latest.json'
    if not ptr.exists(): return None
    try:
        c=root/json.loads(ptr.read_text())['checkpoint']; return c if (c/'adapter').exists() and (c/'state.pt').exists() else None
    except Exception: return None

def save_ckpt(p,mode,model,opt,scaler,step,train_hist,val_hist,best,best_step):
    torch.cuda.synchronize(); root=ckroot(p,mode); root.mkdir(parents=True,exist_ok=True); name=f'step_{step:04d}'; final=root/name; temp=root/f'.tmp_{name}_{os.getpid()}'; shutil.rmtree(temp,ignore_errors=True); shutil.rmtree(final,ignore_errors=True); temp.mkdir(); model.save_pretrained(str(temp/'adapter'),safe_serialization=True)
    torch.save({'step':step,'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'train_history':train_hist,'val_history':val_hist,'best_val':best,'best_step':best_step,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},temp/'state.pt'); temp.rename(final); ptr=root/'latest.json'; prev=None
    if ptr.exists():
        try: prev=json.loads(ptr.read_text()).get('checkpoint')
        except Exception: pass
    atomic_json(ptr,{'checkpoint':name,'step':step})
    if prev and prev!=name: shutil.rmtree(root/prev,ignore_errors=True)
    print(f'[{mode}] checkpoint @ {step}')

def train_student(p,mode,student,val,soft_store=None):
    run=p['students']/mode; run.mkdir(parents=True,exist_ok=True); done=run/'DONE.json'; tcsv=run/'training.csv'; vcsv=run/'validation.csv'
    if done.exists() and tcsv.exists() and vcsv.exists():
        info=json.loads(done.read_text()); print(f'[{mode}] COMPLETE; skip'); return {'train':pd.read_csv(tcsv),'val':pd.read_csv(vcsv),'best_val':float(info['best_val']),'best_step':int(info['best_step'])}
    print('\n'+'='*80+f'\nSTAGE 6: {mode.upper()} STUDENT\n'+'='*80); c=find_ckpt(p,mode)
    if c is None: state=None; adapter=p['initial_adapter']; start=0; th=[]; vh=[]; best=float('inf'); best_step=0
    else: state=torch.load(c/'state.pt',map_location='cpu',weights_only=False); adapter=c/'adapter'; start=int(state['step']); th=list(state['train_history']); vh=list(state['val_history']); best=float(state['best_val']); best_step=int(state['best_step']); print(f'[{mode}] RESUME {start}/100')
    model=load_student(adapter); model.train(); trainable=[x for x in model.parameters() if x.requires_grad]; opt=torch.optim.AdamW(trainable,lr=STUDENT_LR,weight_decay=.01,foreach=False); scaler=torch.amp.GradScaler('cuda')
    if state:
        opt.load_state_dict(state['optimizer'])
        for s in opt.state.values():
            for k,v in list(s.items()):
                if torch.is_tensor(v): s[k]=v.to('cuda')
        scaler.load_state_dict(state['scaler']); random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    else: seed_all(777)
    if start==0: initial=evaluate(model,val); vh.append({'iteration':0,'validation_loss':initial}); best=initial; print(f'[{mode}] initial validation={initial:.6f}')
    for oi in range(start,STUDENT_MAX_STEPS):
        opt.zero_grad(set_to_none=True); hard=soft=combined=0.; domains=[]
        for micro in range(STUDENT_GRAD_ACCUM):
            ex=student[oi*STUDENT_GRAD_ACCUM+micro]; eid=int(ex['example_id']); domains.append(int(ex['domain_id'])); ids=torch.tensor([ex['input_ids']],dtype=torch.long,device='cuda'); mask=torch.ones_like(ids)
            with torch.autocast('cuda',dtype=torch.float16):
                out=model(input_ids=ids,attention_mask=mask,labels=ids,use_cache=False); h=out.loss
                if mode=='baseline': s=None; loss=h
                else: s=kd_loss(soft_store,out.logits,eid); loss=HARD_WEIGHT*h+SOFT_WEIGHT*s
                back=loss/STUDENT_GRAD_ACCUM
            scaler.scale(back).backward(); hard+=float(h.detach().cpu()); combined+=float(loss.detach().cpu()); soft+=0 if s is None else float(s.detach().cpu()); del ex,ids,mask,out,h,loss,back
            if s is not None: del s
        if not np.all(np.bincount(np.asarray(domains),minlength=NUM_DOMAINS)==2): raise RuntimeError('student update not balanced')
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(trainable,GRAD_CLIP); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); step=oi+1; row={'iteration':step,'examples_seen':step*STUDENT_GRAD_ACCUM,'hard_training_loss':hard/STUDENT_GRAD_ACCUM,'combined_training_loss':combined/STUDENT_GRAD_ACCUM};
        if mode=='kd': row['soft_KL']=soft/STUDENT_GRAD_ACCUM
        th.append(row); print(f'[{mode}] {step:3d}/100 CE={row["hard_training_loss"]:.4f}'+(f' KL={row["soft_KL"]:.4f}' if mode=='kd' else ''))
        if step%STUDENT_EVAL_EVERY==0:
            vl=evaluate(model,val); vh.append({'iteration':step,'validation_loss':vl}); print(f'[{mode}] validation={vl:.6f}')
            if vl<best:
                best=vl; best_step=step; bd=run/'best_adapter'; shutil.rmtree(bd,ignore_errors=True); model.save_pretrained(str(bd),safe_serialization=True)
            save_ckpt(p,mode,model,opt,scaler,step,th,vh,best,best_step); model.train()
        elif step%STUDENT_CHECKPOINT_EVERY==0: save_ckpt(p,mode,model,opt,scaler,step,th,vh,best,best_step)
    tdf=pd.DataFrame(th); vdf=pd.DataFrame(vh); tdf.to_csv(tcsv,index=False); vdf.to_csv(vcsv,index=False); atomic_json(done,{'complete':True,'best_val':float(best),'best_step':int(best_step),'final_val':float(vdf.iloc[-1]['validation_loss'])}); del model,opt,scaler,trainable; release_cuda(); print(f'[{mode}] COMPLETE'); return {'train':tdf,'val':vdf,'best_val':best,'best_step':best_step}
