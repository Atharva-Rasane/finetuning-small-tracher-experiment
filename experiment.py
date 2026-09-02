#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'src'
sys.path.insert(0, str(SRC))

import torch
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer
from config import *
from common import paths, seed_all
from data_stage import prepare
from teacher_stage import train_teachers
from score_stage import score_and_select, generate_soft
from student_stage import create_common_adapter, train_student
from report_stage import finalize


def compatibility_check():
    print('=' * 80)
    print('ENVIRONMENT / MODEL COMPATIBILITY')
    print('=' * 80)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {name} ({memory:.2f} GiB)')
    print(f'PyTorch: {torch.__version__} | CUDA runtime: {torch.version.cuda}')

    tt = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    st = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tt.get_vocab() != st.get_vocab():
        raise RuntimeError('teacher/student token->ID mappings differ; direct token-level KD is invalid')

    tc = AutoConfig.from_pretrained(TEACHER_MODEL)
    sc = AutoConfig.from_pretrained(STUDENT_MODEL)
    if tc.vocab_size != sc.vocab_size:
        raise RuntimeError('teacher/student model vocab sizes differ')
    if BLOCK_SIZE > min(tc.n_positions, sc.n_positions):
        raise RuntimeError('BLOCK_SIZE exceeds model context')

    separator = st.encode('\n\n', add_special_tokens=False)
    if not separator or max(separator) >= tc.vocab_size:
        raise RuntimeError('safe ordinary-token file separator could not be constructed')

    print(f'Tokenizer compatibility: PASS (vocab={tc.vocab_size})')
    return st, tc.vocab_size, separator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', default=os.environ.get('EXPERIMENT_STATE_DIR', str(HERE / 'state')))
    args = parser.parse_args()
    p = paths(args.state_dir)

    if p['finished'].exists():
        print('EXPERIMENT FINISHED')
        return 0

    seed_all(SEED)
    tokenizer, vocab_size, separator = compatibility_check()

    student, val = prepare(p, tokenizer, vocab_size, separator)
    train_teachers(p)
    winners = score_and_select(p, student)
    generate_soft(p, student, winners)
    create_common_adapter(p)

    soft_store = torch.load(p['soft_file'], map_location='cpu', weights_only=False)
    baseline = train_student(p, 'baseline', student, val, soft_store=None)
    kd = train_student(p, 'kd', student, val, soft_store=soft_store)
    finalize(p, baseline, kd)

    if not p['finished'].exists():
        raise RuntimeError('finalization returned without finished marker')
    print('EXPERIMENT FINISHED')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
