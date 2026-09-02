import hashlib, gc, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from config import *
from common import atomic_json


def save_dataset_atomic(dataset, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.parent/f'.tmp_{path.name}_{os.getpid()}'
    shutil.rmtree(temp,ignore_errors=True); dataset.save_to_disk(str(temp))
    if path.exists(): shutil.rmtree(path,ignore_errors=True)
    temp.rename(path)

def unique_indices(raw):
    seen=set(); out=[]
    for i in range(len(raw)):
        text=raw[i].get('content','')
        if not isinstance(text,str) or not text.strip(): continue
        h=hashlib.sha1(text.encode('utf-8',errors='ignore')).digest()
        if h in seen: continue
        seen.add(h); out.append(i)
    return np.asarray(out,dtype=np.int64)

def balanced_bins(raw, indices):
    if 'size' in raw.column_names:
        sizes=np.asarray(raw['size'],dtype=np.int64); weights=np.maximum(sizes[indices],1)
    else:
        weights=np.asarray([max(len(raw[int(i)].get('content','')),1) for i in indices],dtype=np.int64)
    order=np.argsort(weights)[::-1]; bins=[[] for _ in range(NUM_TEACHERS)]; totals=np.zeros(NUM_TEACHERS,dtype=np.int64)
    for pos in order:
        t=int(np.argmin(totals)); idx=int(indices[pos]); bins[t].append(idx); totals[t]+=int(weights[pos])
    return [np.asarray(x,dtype=np.int64) for x in bins]

def pack(raw, file_indices, needed, tokenizer, vocab_size, separator):
    blocks=[]; buffer=[]
    for idx in file_indices:
        text=raw[int(idx)].get('content','')
        if not isinstance(text,str) or not text.strip(): continue
        ids=tokenizer.encode(text,add_special_tokens=False)
        if ids and max(ids)>=vocab_size: raise RuntimeError(f'token id {max(ids)} >= vocab {vocab_size}')
        buffer.extend(ids); buffer.extend(separator)
        take=min(len(buffer)//BLOCK_SIZE, needed-len(blocks))
        for b in range(take):
            s=b*BLOCK_SIZE; blocks.append(buffer[s:s+BLOCK_SIZE])
        consumed=take*BLOCK_SIZE
        if consumed: buffer=buffer[consumed:]
        if len(blocks)>=needed: return blocks
    return blocks

def prepare(p, tokenizer, vocab_size, separator):
    if p['prepared'].exists() and p['student_data'].exists() and p['val_data'].exists() and all((p['data']/f'teacher_{i}').exists() for i in range(NUM_TEACHERS)):
        print('[data] already prepared')
        return load_from_disk(str(p['student_data'])), load_from_disk(str(p['val_data']))
    print('\n'+'='*80+'\nSTAGE 1: DATA PREPARATION\n'+'='*80)
    for d,(folder,name) in enumerate(DOMAINS):
        marker=p['parts']/f'domain_{d}_DONE.json'
        expected=[p['parts']/f'teacher_{t}_domain_{d}' for t in range(NUM_TEACHERS)]+[p['parts']/f'student_domain_{d}',p['parts']/f'validation_domain_{d}']
        if marker.exists() and all(x.exists() for x in expected):
            print(f'[data] {name}: cached'); continue
        raw=load_dataset(DATASET_NAME,data_dir=f'data/{folder}',split='train')
        unique=unique_indices(raw); rng=np.random.default_rng(SEED+d*10000); order=rng.permutation(unique); n=len(order)
        nt=int(n*TEACHER_FILE_FRAC); ns=int(n*STUDENT_FILE_FRAC)
        tpool=order[:nt]; spool=order[nt:nt+ns]; vpool=order[nt+ns:]; bins=balanced_bins(raw,tpool)
        print(f'[data] {name}: raw={len(raw):,} unique={n:,}')
        for t,files in enumerate(bins):
            out=p['parts']/f'teacher_{t}_domain_{d}'
            if out.exists(): continue
            blocks=pack(raw,files,TEACHER_BLOCKS_PER_DOMAIN,tokenizer,vocab_size,separator)
            if len(blocks)!=TEACHER_BLOCKS_PER_DOMAIN: raise RuntimeError(f'T{t}/{name}: only {len(blocks)} blocks')
            save_dataset_atomic(Dataset.from_dict({'input_ids':blocks,'domain_id':[d]*len(blocks)}),out); print(f'[data] T{t}/{name}: {len(blocks)} blocks')
        sout=p['parts']/f'student_domain_{d}'
        if not sout.exists():
            blocks=pack(raw,spool,STUDENT_BLOCKS_PER_DOMAIN,tokenizer,vocab_size,separator)
            if len(blocks)!=STUDENT_BLOCKS_PER_DOMAIN: raise RuntimeError(f'{name}: insufficient student blocks')
            save_dataset_atomic(Dataset.from_dict({'input_ids':blocks,'domain_id':[d]*len(blocks)}),sout)
        vout=p['parts']/f'validation_domain_{d}'
        if not vout.exists():
            blocks=pack(raw,vpool,VAL_BLOCKS_PER_DOMAIN,tokenizer,vocab_size,separator)
            if len(blocks)!=VAL_BLOCKS_PER_DOMAIN: raise RuntimeError(f'{name}: insufficient validation blocks')
            save_dataset_atomic(Dataset.from_dict({'input_ids':blocks,'domain_id':[d]*len(blocks)}),vout)
        atomic_json(marker,{'domain':name,'raw_files':len(raw),'unique_files':n,'teacher_pool_files':len(tpool),'student_pool_files':len(spool),'validation_pool_files':len(vpool)})
        del raw,unique,order,tpool,spool,vpool,bins; gc.collect()
    for t in range(NUM_TEACHERS):
        out=p['data']/f'teacher_{t}'
        if out.exists(): continue
        parts=[load_from_disk(str(p['parts']/f'teacher_{t}_domain_{d}')) for d in range(NUM_DOMAINS)]
        ds=concatenate_datasets(parts); counts=np.bincount(np.asarray(ds['domain_id']),minlength=NUM_DOMAINS)
        if not np.all(counts==TEACHER_BLOCKS_PER_DOMAIN): raise RuntimeError(f'T{t} balance failed')
        save_dataset_atomic(ds.shuffle(seed=SEED+50000+t),out); print(f'[data] teacher {t}: {len(ds):,} blocks')
    student_parts=[]
    for d in range(NUM_DOMAINS): student_parts.append(load_from_disk(str(p['parts']/f'student_domain_{d}')).shuffle(seed=SEED+60000+d))
    ordered=[]
    for step in range(STUDENT_MAX_STEPS):
        for d in range(NUM_DOMAINS):
            for j in range(step*2,step*2+2): ordered.append(student_parts[d][j])
    student=Dataset.from_dict({'input_ids':[x['input_ids'] for x in ordered],'domain_id':[x['domain_id'] for x in ordered],'example_id':list(range(STUDENT_EXAMPLES))})
    for step in range(STUDENT_MAX_STEPS):
        ids=np.asarray(student[step*STUDENT_GRAD_ACCUM:(step+1)*STUDENT_GRAD_ACCUM]['domain_id']); counts=np.bincount(ids,minlength=NUM_DOMAINS)
        if not np.all(counts==2): raise RuntimeError('student batch balance failure')
    save_dataset_atomic(student,p['student_data'])
    val_parts=[load_from_disk(str(p['parts']/f'validation_domain_{d}')) for d in range(NUM_DOMAINS)]
    val=concatenate_datasets(val_parts).shuffle(seed=SEED+70000); save_dataset_atomic(val,p['val_data'])
    audit=[]
    for t in range(NUM_TEACHERS):
        ds=load_from_disk(str(p['data']/f'teacher_{t}')); counts=np.bincount(np.asarray(ds['domain_id']),minlength=NUM_DOMAINS)
        for d,(_,name) in enumerate(DOMAINS): audit.append({'teacher':t,'domain':name,'blocks':int(counts[d]),'tokens':int(counts[d]*BLOCK_SIZE)})
    pd.DataFrame(audit).to_csv(p['root']/'teacher_domain_balance.csv',index=False)
    atomic_json(p['prepared'],{'balanced':True,'blocks_per_domain_per_teacher':TEACHER_BLOCKS_PER_DOMAIN,'tokens_per_teacher':TEACHER_TOKENS_TOTAL,'total_teacher_tokens':TOTAL_TEACHER_TOKENS,'student_examples':STUDENT_EXAMPLES,'validation_examples':VALIDATION_EXAMPLES})
    print('[data] COMPLETE')
    return student,val
