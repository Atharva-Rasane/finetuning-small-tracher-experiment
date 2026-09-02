# Balanced multi-teacher fine-tuning experiment

This repository runs the complete CodeParrot teacher-selection / knowledge-distillation experiment on a single NVIDIA T4 VM.

The intended usage is deliberately simple:

```bash
git clone https://github.com/Atharva-Rasane/finetuning-small-tracher-experiment.git
cd finetuning-small-tracher-experiment
./run.sh
```

`run.sh` creates an isolated Python environment, installs the pinned CUDA/PyTorch/Hugging Face dependencies, validates the GPU, and starts or resumes the experiment. If the experiment has already completed, it prints:

```text
EXPERIMENT FINISHED
```

and exits without loading models or datasets.

## Recommended VM

For Google Cloud, the recommended first run is:

- 1 x NVIDIA T4 (16 GB VRAM)
- `n1-standard-8` (8 vCPUs, 30 GB RAM), or another VM with roughly 8 vCPUs / 32 GB RAM
- 200 GB persistent `pd-balanced` or SSD storage
- Ubuntu 22.04 GPU-enabled / Deep Learning VM image
- NVIDIA driver installed and visible through `nvidia-smi`
- standard VM rather than Spot for the first full experiment

The experiment itself does not require a local CUDA toolkit. `run.sh` creates a virtual environment and installs the PyTorch 2.7.0 CUDA 11.8 wheel. The host only needs a sufficiently new working NVIDIA driver.

Do not use Local SSD as the only place for experiment state. The default `state/` directory is deliberately persistent relative to the repository. If you attach a separate persistent data disk, point the experiment at it:

```bash
EXPERIMENT_STATE_DIR=/mnt/disks/experiment/state ./run.sh
```

This is preferable for Spot/preemptible VMs or if you plan to replace the VM while keeping the experiment disk.

## Experiment design

Models:

- 8 teachers: `codeparrot/codeparrot-small` (~110M parameters)
- student: `codeparrot/codeparrot` (~1.5B parameters)
- dataset: `bigcode/the-stack-smol-xl`

Domains:

- Rust
- Go
- Java
- C++
- C#
- TypeScript
- Shell
- SQL

Every teacher receives the same domain distribution. Each teacher gets exactly 2,500 packed 256-token blocks from every domain:

```text
2,500 blocks/domain
x 8 domains
= 20,000 blocks/teacher
= 5,120,000 fine-tuning tokens/teacher

x 8 teachers
= 40,960,000 total teacher fine-tuning tokens
```

Source files are split before tokenization. Within each language, teacher file sets are mutually disjoint. Student-training files and validation files are held out from all eight teachers.

The student comparison uses 1,600 balanced held-out sequences. There are 100 optimizer updates with gradient accumulation 16. Every effective update contains exactly two examples from each of the eight domains.

The teachers are fully trained before scoring or soft-label generation begins.

## Pipeline

The stages run strictly in this order:

```text
1. Prepare and audit balanced data partitions
2. Fully fine-tune Teacher 0 ... Teacher 7
3. Score the untouched 110M base and all eight final teachers
4. Choose the lowest-NLL teacher for each held-out student sequence
5. Generate top-32 temperature-softened labels using final teacher weights
6. Create one common 1.5B LoRA initialization
7. Train 1.5B baseline for 100 updates
8. Train 1.5B KD student for 100 updates
9. Produce comparison CSV/SVG and mark the experiment complete
```

The baseline and KD student start from exactly the same saved LoRA initialization, consume exactly the same student sequences in the same order, and use the same optimizer settings. The only intended difference is the loss function.

## Resume / fault tolerance

Durable state lives under `state/` by default.

Teacher training saves a full resumable checkpoint every 100 optimizer updates, including model weights, Adam state, LR scheduler, GradScaler, progress position, and RNG state. Only the most recent teacher checkpoint is retained. A completed teacher is never retrained.

Scoring results are cached independently for the base model and each of the eight teachers. Soft labels are generated as one durable shard per winning teacher, so completed shards are not regenerated.

Baseline and KD student runs checkpoint every 5 optimizer updates and after every validation. They restore adapter weights, optimizer, GradScaler, histories, and RNG states.

`run.sh` also acts as a process-level CUDA supervisor. If `experiment.py` exits unexpectedly, it starts a fresh Python/CUDA process and resumes from the latest durable checkpoint. It performs at most five consecutive automatic restarts so a deterministic coding/configuration error does not create an infinite loop.

You can always rerun:

```bash
./run.sh
```

after fixing a VM issue, rebooting, or reconnecting.

## Important state files

```text
state/
├── datasets/
├── teachers/
│   ├── teacher_0/
│   │   ├── final_model/
│   │   └── DONE.json
│   └── ...
├── scores/
├── softlabels/
├── students/
│   ├── baseline/
│   └── kd/
├── reports/
├── logs/
└── EXPERIMENT_FINISHED.json
```

`state/EXPERIMENT_FINISHED.json` is written only after the complete baseline/KD report has been generated.

## Main diagnostics

Before interpreting student KD, inspect the teacher diagnostics. The experiment measures:

```text
Fine-tuning gain
= NLL(untuned 110M base) - NLL(mean fine-tuned teacher)

Oracle diversity gain
= NLL(mean fine-tuned teacher) - NLL(best teacher per sequence)
```

It also records teacher winner counts, winner entropy, best-vs-second-best NLL margin, and per-domain teacher NLLs.

The student result compares held-out validation cross-entropy. Baseline hard CE and KD hard CE are directly comparable; the KD combined CE+KL objective is not compared numerically against baseline CE.

## Outputs

After completion, the important files are:

```text
state/teacher_domain_balance.csv
state/teacher_training.csv
state/teacher_domain_nll.csv
state/scores/diagnostics.json
state/scores/oracle_selection.csv
state/students/baseline/training.csv
state/students/baseline/validation.csv
state/students/kd/training.csv
state/students/kd/validation.csv
state/reports/final_results.csv
state/reports/baseline_vs_kd.svg
state/reports/summary.json
state/EXPERIMENT_FINISHED.json
```

All generated state is ignored by Git via `.gitignore`.

## Logs

Every supervisor attempt is written to:

```text
state/logs/
```

The console output is also streamed live with `tee`.

## Dependency policy

The experiment uses a pinned, known-compatible stack rather than installing whichever Hugging Face release happens to be newest when the VM starts. This also avoids the `TrainingArguments` API mismatch encountered in the notebook version. Teacher and student optimization are implemented as explicit PyTorch loops; Hugging Face `Trainer` is not used.
