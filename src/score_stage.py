import json, math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from config import *
from common import atomic_json, atomic_numpy, atomic_torch, release_cuda
from teacher_stage import load_teacher

def collate(examples):
    ids=torch.tensor([x['input_ids'] for x in examples],dtype=torch.long); out={'input_ids':ids,'attention_mask':torch.ones_like(ids),'labels':ids.clone()}
    if 'example_id' in examples[0]: out['example_id']=torch.tensor([x['example_id'] for x in examples],dtype=torch.long)
    if 'domain_id' in examples[0]: out['domain_id']=torch.tensor([x['domain_id'] for x in examples],dtype=torch.long)
    return out

@torch.inference_mode()
def seq_nll(logits,labels):
    x=logits[:,:-1,:].float(); y=labels[:,1:]; gold=x.gather(-1,y.unsqueeze(-1)).squeeze(-1); return (torch.logsumexp(x,dim=-1)-gold).mean(dim=1)

def score_model(source,cache,label,student):
    if cache.exists(): print(f'[score] {label}: cached'); return np.load(cache)
    print(f'[score] {label}'); model=load_teacher(source,torch.float16).to('cuda'); model.eval(); loader=DataLoader(student,batch_size=8,shuffle=False,collate_fn=collate,num_workers=2,pin_memory=True); scores=np.empty(STUDENT_EXAMPLES,dtype=np.float32); done=0
    with torch.inference_mode():
        for batch in loader:
            eids=batch['example_id'].numpy(); ids=batch['input_ids'].to('cuda',non_blocking=True); mask=batch['attention_mask'].to('cuda',non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16): out=model(input_ids=ids,attention_mask=mask,use_cache=False)
            vals=seq_nll(out.logits,ids); scores[eids]=vals.cpu().numpy(); done+=len(eids)
            if done%400<len(eids): print(f'[score] {label}: {done}/{STUDENT_EXAMPLES}')
            del ids,mask,out,vals
    torch.cuda.synchronize(); atomic_numpy(scores,cache); del model,loader; release_cuda(); return scores

def score_and_select(p,student):
    print('\n'+'='*80+'\nSTAGE 3: FINAL-TEACHER SCORING / ORACLE\n'+'='*80)
    base=score_model(TEACHER_MODEL,p['scores']/'base_110m_nll.npy','untuned 110M',student); vectors=[]
    for t in range(NUM_TEACHERS): vectors.append(score_model(p['teachers']/f'teacher_{t}'/'final_model',p['scores']/f'teacher_{t}_nll.npy',f'teacher {t}',student))
    matrix=np.stack(vectors,axis=1); atomic_numpy(matrix,p['scores']/'teacher_nll_matrix.npy'); winners=matrix.argmin(axis=1); best=matrix[np.arange(STUDENT_EXAMPLES),winners]; mean=matrix.mean(axis=1); sorted_nll=np.sort(matrix,axis=1); second=sorted_nll[:,1]; domains=np.asarray(student['domain_id'],dtype=np.int64)
    counts=np.bincount(winners,minlength=NUM_TEACHERS); probs=counts/counts.sum(); nz=probs[probs>0]; ent=float(-(nz*np.log2(nz)).sum()); diag={'untuned_base_mean_nll':float(base.mean()),'mean_finetuned_teacher_nll':float(mean.mean()),'oracle_best_teacher_nll':float(best.mean()),'fine_tuning_gain_base_minus_mean_teacher':float(base.mean()-mean.mean()),'oracle_diversity_gain_mean_minus_best':float(mean.mean()-best.mean()),'mean_best_vs_second_margin':float((second-best).mean()),'winner_counts':counts.tolist(),'winner_entropy_bits':ent,'winner_entropy_normalized':ent/math.log2(NUM_TEACHERS)}; atomic_json(p['scores']/'diagnostics.json',diag); atomic_numpy(winners,p['scores']/'winning_teacher.npy')
    pd.DataFrame({'example_id':np.arange(STUDENT_EXAMPLES),'domain':[DOMAINS[i][1] for i in domains],'base_nll':base,'mean_teacher_nll':mean,'best_teacher':winners,'best_teacher_nll':best,'selection_margin':second-best,'oracle_gain_over_mean':mean-best}).to_csv(p['scores']/'oracle_selection.csv',index=False)
    rows=[]
    for d,(_,name) in enumerate(DOMAINS):
        m=domains==d; row={'domain':name,'base':float(base[m].mean()),'mean_teacher':float(mean[m].mean()),'oracle':float(best[m].mean())}; row.update({f'T{t}':float(matrix[m,t].mean()) for t in range(NUM_TEACHERS)}); rows.append(row)
    pd.DataFrame(rows).to_csv(p['root']/'teacher_domain_nll.csv',index=False); print(json.dumps(diag,indent=2)); return winners

def generate_soft(p,student,winners):
    if p['soft_file'].exists(): print('[soft] merged cache COMPLETE'); return
    print('\n'+'='*80+'\nSTAGE 4: SOFT LABELS FROM FULLY TRAINED TEACHERS\n'+'='*80); shards=p['soft']/'shards'; shards.mkdir(parents=True,exist_ok=True)
    for t in range(NUM_TEACHERS):
        file=shards/f'teacher_{t}.pt'
        if file.exists(): print(f'[soft] T{t}: cached'); continue
        selected=np.where(winners==t)[0]; print(f'[soft] T{t}: {len(selected)} examples')
        if len(selected)==0: atomic_torch({'ids':torch.empty(0,dtype=torch.int32)},file); continue
        subset=student.select(selected.tolist()); loader=DataLoader(subset,batch_size=4,shuffle=False,collate_fn=collate,num_workers=2,pin_memory=True); model=load_teacher(p['teachers']/f'teacher_{t}'/'final_model',torch.float16).to('cuda'); model.eval(); A=[];B=[];C=[];D=[]
        with torch.inference_mode():
            for batch in loader:
                e=batch['example_id'].to(torch.int32); ids=batch['input_ids'].to('cuda',non_blocking=True); mask=batch['attention_mask'].to('cuda',non_blocking=True)
                with torch.autocast('cuda',dtype=torch.float16): out=model(input_ids=ids,attention_mask=mask,use_cache=False)
                logits=out.logits[:,:-1,:]; vals,inds=torch.topk(logits,k=TOP_K,dim=-1); z=torch.logsumexp(logits.float()/TEMPERATURE,dim=-1); A.append(e); B.append(inds.cpu().to(torch.int32)); C.append(vals.cpu().to(torch.float16)); D.append(z.cpu().to(torch.float32)); del ids,mask,out,logits,vals,inds,z
        torch.cuda.synchronize(); atomic_torch({'ids':torch.cat(A),'top_indices':torch.cat(B),'top_logits':torch.cat(C),'log_z_T':torch.cat(D)},file); del model,loader,subset,A,B,C,D; release_cuda()
    pos=BLOCK_SIZE-1; inds=torch.empty((STUDENT_EXAMPLES,pos,TOP_K),dtype=torch.int32); logits=torch.empty((STUDENT_EXAMPLES,pos,TOP_K),dtype=torch.float16); z=torch.empty((STUDENT_EXAMPLES,pos),dtype=torch.float32); filled=torch.zeros(STUDENT_EXAMPLES,dtype=torch.bool)
    for t in range(NUM_TEACHERS):
        s=torch.load(shards/f'teacher_{t}.pt',map_location='cpu',weights_only=False); ids=s['ids'].long()
        if len(ids): inds[ids]=s['top_indices']; logits[ids]=s['top_logits']; z[ids]=s['log_z_T']; filled[ids]=True
    if not bool(filled.all()): raise RuntimeError('incomplete soft labels')
    atomic_torch({'metadata':{'examples':STUDENT_EXAMPLES,'block_size':BLOCK_SIZE,'top_k':TOP_K,'temperature':TEMPERATURE,'teachers_fully_trained_before_labeling':True,'balanced_teachers':True},'top_indices':inds,'top_logits':logits,'log_z_T':z,'selected_teacher':torch.from_numpy(winners).long(),'domain_id':torch.tensor(student['domain_id'],dtype=torch.long)},p['soft_file']); print('[soft] COMPLETE')
