# CheXInstruct-MedGemma Foundation Finetuning Framework

A framework for instruction fine-tuning of **MedGemma (Gemma-3-4b-it)** on the **CheXInstruct** benchmark dataset for chest X-ray understanding, using multi-node distributed training with DeepSpeed ZeRO-3 and LoRA on the Alvis HPC cluster.

---

## Table of Contents

1. [Dataset: CheXInstruct](#dataset-chexinstruct)
2. [Model & Architecture](#model--architecture)
3. [Training Setup](#training-setup)
4. [Repository Structure](#repository-structure)
5. [Data Pipeline](#data-pipeline)
6. [Evaluation](#evaluation)
7. [Getting Started](#getting-started)

---

## Dataset: CheXInstruct

### Overview

**CheXInstruct** is a large-scale, multi-task instruction-following benchmark for chest X-ray analysis. It aggregates QA pairs and annotations from a diverse collection of public radiology datasets, covering 24 distinct clinical and research tasks. Each sample consists of one or more chest X-ray images paired with a natural language instruction and a reference answer, formatted into a conversation suitable for vision-language model fine-tuning.

The dataset is distributed as `.pkl` files (train/val/test splits) and converted to HuggingFace Parquet format for efficient loading during training.

### Source Datasets

CheXInstruct aggregates data from the following publicly available chest X-ray corpora:

| Dataset | Description |
|---|---|
| **MIMIC-CXR-JPG** | Large-scale radiology report dataset (PhysioNet, requires credentialed access) |
| **CheXpert** | Stanford chest X-ray dataset with structured labels and free-text reports |
| **VinDr-CXR** | Vietnamese multi-disease chest X-ray dataset with bounding box annotations |
| **BRAX** | Brazilian radiology dataset with radiologist reports |
| **OpenI** | Indiana University chest X-ray collection with paired reports |
| **COVIDx-CXR4** | COVID-19 chest X-ray dataset for detection and classification |
| **MS-CXR** | Phrase-grounding dataset mapping report sentences to image bounding boxes |
| **MS-CXR-T** | Temporal extension of MS-CXR for progression analysis |
| **Rad-ReStruct** | Structured radiology report dataset |
| **RadNLI** | Natural language inference dataset for radiology reports |
| **PMC-VQA** | PubMed Central medical visual question answering |
| **MIMIC-Diff-VQA** | Difference VQA dataset based on MIMIC-CXR temporal pairs |
| **CXR-LT** | Long-tail chest X-ray classification dataset |
| **Object-CXR** | Foreign object detection in chest X-rays |
| **NLM-TB** | NLM tuberculosis chest X-ray dataset |
| **Rex-Gradient-160K** | Large-scale radiology explanation dataset |

### Supported Tasks (24 categories)

CheXInstruct spans a comprehensive set of clinical imaging tasks, grouped by type:

#### Classification & Detection
| Task | Description |
|---|---|
| **Image Classification** | Identify and classify pathological conditions from a chest X-ray |
| **Temporal Image Classification** | Classify disease progression/improvement across sequential images |
| **View Classification** | Identify the radiographic projection type (PA, AP, lateral) |
| **Abnormality Detection** | Detect and localize pathological findings |
| **Foreign Object Detection** | Identify non-anatomical structures or medical devices |

#### Grounding & Localization
| Task | Description |
|---|---|
| **Abnormality Grounding** | Provide spatial localization for detected abnormalities |
| **Phrase Grounding** | Link textual descriptions to specific image regions |
| **Grounded Captioning** | Generate captions with anatomical spatial grounding |
| **Grounded Diagnosis** | Provide diagnoses with spatial anatomical localization |
| **Grounded Phrase Extraction** | Extract key phrases and ground them to image regions |
| **Pneumothorax Segmentation** | Delineate pneumothorax regions |

#### Report Generation
| Task | Description |
|---|---|
| **Findings Generation** | Generate comprehensive radiological findings from the image |
| **Findings Generation with Indication** | Generate findings conditioned on clinical indication |
| **Impression Generation** | Synthesize findings into a diagnostic impression |
| **Impression Generation with Indication** | Generate impressions incorporating clinical context |
| **Progression Findings Generation** | Describe temporal changes between sequential X-rays |
| **Progression Impression Generation** | Summarize disease progression into a clinical impression |
| **Findings Summarization** | Condense detailed findings into concise summaries |

#### Visual Question Answering
| Task | Description |
|---|---|
| **Open-Ended VQA** | Answer free-form questions about a chest X-ray |
| **Close-Ended VQA** | Answer specific questions with precise, direct responses |
| **Difference VQA** | Answer questions about differences between two images |
| **Text QA** | Answer questions based on radiological text and reports |
| **Report Evaluation** | Assess radiological report quality and completeness |

#### Natural Language & Clinical Reasoning
| Task | Description |
|---|---|
| **Natural Language Explanation** | Explain chest X-ray findings in accessible natural language |
| **Natural Language Inference** | Perform logical reasoning on radiological statements |
| **Temporal Sentence Similarity** | Analyze similarity between radiological descriptions over time |
| **Named Entity Recognition** | Extract medical entities from radiology reports |

### Data Format

Each sample in the raw CheXInstruct dataset has the following structure:

```json
{
  "unique_id": "DATASET_NAME[Task Name]_identifier",
  "image_path": ["/path/to/image.jpg"],
  "qa_pair": [
    { "q": "Describe the findings in this chest X-ray.", "a": "The lungs are clear..." }
  ]
}
```

The `unique_id` field encodes the source dataset and task type using bracket notation (e.g., `[Findings Generation]`), which is used during data loading to route samples to the correct task-specific system prompt.

### Conversation Format

During preprocessing, each QA pair is converted into a multi-turn conversation with a task-specific **system prompt** randomly sampled from a curated pool:

```
[system]  "You are a radiology AI assistant that generates comprehensive radiological findings..."
[user]    <chest_x_ray_image> + instruction text
[assistant] reference answer
```

This format is compatible with Gemma-3's instruction-tuning template. Equivalent formatters are provided for **LLaVA 1.5** and **Qwen2.5-VL**.

### Dataset Statistics

The dataset is split into train, validation, and test sets. A QA sampling rate of `0.01` (1%) is applied per sample during HF dataset creation to manage dataset size while preserving task diversity.

| Split | File |
|---|---|
| Train | `data_chexinstruct/data_train_chexinstruct.pkl` |
| Validation | `data_chexinstruct/data_val_chexinstruct.pkl` |
| Test | `data_chexinstruct/data_test_chexinstruct.pkl` |

After formatting, the HuggingFace Parquet datasets are stored under:

```
data_chexinstruct/
├── hf_parquet_gemma_format/gemma/         # Gemma conversation format
├── hf_parquet_gemma_format/gemma_findings/ # Findings task only
└── hf_parquet_gemma_format/gemma_3_findings_tok/ # Tokenized (ready for training)
```

---

## Model & Architecture

- **Base model**: [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) (MedGemma)
- **Fine-tuning strategy**: LoRA (Low-Rank Adaptation) with Gaussian initialization
  - Rank: 64 | Alpha: 64
- **Precision**: bfloat16
- **Kernel optimization**: [Liger Kernel](https://github.com/linkedin/Liger-Kernel) for Gemma-3 (`apply_liger_kernel_to_gemma3_text`)
- **Custom forward pass**: monkey-patched Gemma-3 forward for compatibility with multi-image inputs

---

## Training Setup

Training runs on the **Alvis HPC cluster** (NAISS, Sweden) using SLURM.

| Parameter | Value |
|---|---|
| Nodes | 2 |
| GPUs | 4× A100 per node (8 total) |
| Parallelism | DeepSpeed ZeRO-3 + Accelerate |
| Batch size (per device) | 4 |
| Gradient accumulation | 4 steps (effective batch: 128) |
| Learning rate | 2e-4 |
| LR scheduler | `cosine_with_restarts` |
| Warmup ratio | 0.05 |
| Weight decay | 0.0001 |
| Epochs | 3 |
| Max sequence length | 1500 tokens |
| Gradient checkpointing | Enabled |
| Experiment tracking | Weights & Biases (wandb) |

Launch command (SLURM):
```bash
sbatch accelerated_chexinstruct_gemma3_multinode_z3.sh
```

---

## Repository Structure

```
.
├── src/
│   ├── dataset/
│   │   ├── chexinstruct/          # Dataset creation and formatting scripts
│   │   │   ├── createHFDataset.py # Convert .pkl splits to HF Parquet format
│   │   │   ├── format_chexinstruct.py  # Tokenize formatted dataset
│   │   │   └── FORMAT_DATA.md     # Data formatting usage guide
│   │   ├── formatters.py          # Conversation formatters (Gemma, LLaVA, Qwen)
│   │   ├── system_prompt.py       # Task-specific radiology system prompts
│   │   └── util_data.py           # Parquet load/save utilities
│   ├── finetune/
│   │   ├── finetune_accelerated_v2.py  # Main training script (Accelerate + DeepSpeed)
│   │   └── monkey_patch_forward.py    # Gemma-3 forward pass patch
│   ├── models/
│   │   ├── Gemma3.py              # Gemma-3 model wrapper and collator
│   │   ├── peft.py                # LoRA / PEFT configuration
│   │   └── VisionLanguage.py      # Base VLM interface
│   ├── benchmark/
│   │   └── benchmark_VLM.py       # VLM inference benchmarking
│   ├── inference/
│   │   └── generate_reports.py    # Report generation for evaluation
│   ├── download/                  # Download scripts for each source dataset
│   └── preprocess/                # Dataset-specific preprocessing scripts
├── deepspeed/
│   ├── ds_zero3_config.yaml       # DeepSpeed ZeRO-3 config (used in training)
│   └── ds_zero2_config_bf16.json  # Alternative ZeRO-2 config
├── modules/                       # Git submodules
│   ├── ReXrank/                   # Radiology report ranking evaluation
│   ├── GREEN/                     # GREEN score for report evaluation
│   └── cxr-metrics/               # CXR-specific NLP metrics
└── accelerated_chexinstruct_gemma3_multinode_z3.sh  # Main SLURM launch script
```

---

## Downloading CheXInstruct

### 1. Download the CheXInstruct annotation files

CheXInstruct is hosted on HuggingFace. Download the annotation pickle files (train/val/test splits) using the HuggingFace CLI:

```bash
# Authenticate first (requires HuggingFace account)
huggingface-cli login

# Download CheXInstruct annotation splits
huggingface-cli download jbdel/chexinstruct \
    --repo-type dataset \
    --local-dir data_chexinstruct
```

This will place three files in `data_chexinstruct/`:

```
data_chexinstruct/
├── data_train_chexinstruct.pkl
├── data_val_chexinstruct.pkl
└── data_test_chexinstruct.pkl
```

> **Note**: The `.pkl` files contain metadata (image paths, QA pairs, task labels) but **not** the images themselves. Images must be downloaded separately from each source dataset.

### 2. Download source images

Each source dataset requires a separate download. Scripts are in `src/download/`. Most require free registration on the respective platforms (PhysioNet, Kaggle, etc.).

```bash
# MIMIC-CXR-JPG (requires PhysioNet credentialed access)
bash src/download/MIMIC-CXR-JPG.sh

# CheXpert (requires Stanford registration)
bash src/download/download_chest_x_ray_data.sh

# VinDr-CXR (requires Kaggle API)
bash src/download/vindr_cxr/download_vindr_cxr.sh

# OpenI (public)
bash src/download/OpenI.sh

# COVIDx-CXR4 (public)
bash src/download/COVIDx-CXR4.bash

# BRAX (public)
python src/download/brax/downloadBrax.py

# Rex-Gradient-160K (requires HF token)
bash src/download/rex-gradient160k/download_rexgrad160k.sh
```

### Source datasets on disk

All images are resized to 512×512 px and stored as PNG/JPG. The table below describes the dataset layout as downloaded.

| Dataset | Access | Key files / folders | Format |
|---|---|---|---|
| **MIMIC-CXR-JPG** | PhysioNet (credentialed) | `mimic-cxr/files_512/`, `mimic-cxr-2.0.0-split.csv`, `mimic-cxr-reports/` | JPG 512px |
| **CheXpert** | Stanford (free registration) | `chexpert-public/train-512/`, `valid-512/`, `test-512/`, `train.csv`, `valid.csv` | JPG 512px |
| **VinDr-CXR** | Kaggle / PhysioNet | `vindr-cxr/train_png/`, `test_png/`, `annotations_train.csv`, `annotations_test.csv` | PNG 512px |
| **BRAX** | Public (GCS) | `brax/images-512/`, `brax_train_both_cleaned.csv`, `brax_test_both_cleaned.csv` | JPG 512px |
| **OpenI** | Public (NLM) | `openi/images-512/`, `indiana_reports.csv`, `indiana_projections.csv` | PNG 512px |
| **ChestX-ray14** | Public (NIH) | `chestxray14/images-512/`, `Data_Entry_2017.csv`, `BBox_List_2017.csv` | PNG 512px |
| **MS-CXR** | PhysioNet (credentialed) | `ms-cxr/1.0.0/`, `MS_CXR_Local_Alignment_v1.1.0.csv` | JPG 512px |
| **MS-CXR-T** | PhysioNet (credentialed) | `ms-cxr-t/`, `MS_CXR_T_temporal_image_classification_v1.0.0.csv`, `MS_CXR_T_temporal_sentence_similarity_v1.0.0.csv` | JPG 512px |
| **COVIDx-CXR-3** | Public (Kaggle) | `covidx-cxr-3/images-512/`, `train.txt`, `val.txt`, `test.txt` | PNG 512px |
| **CXR-LT** | Public (MIMIC subset) | `cxr-lt/cxr-lt-2023/`, `cxr-lt-2024/`, `train.csv`, `test.csv` | JPG 512px |
| **Object-CXR** | Public | `object-cxr/train/`, `dev/`, `train.csv`, `dev.csv` | JPG 512px |
| **NLM-TB** | Public (NLM) | `nlm-tb/TB_Chest_Radiography_Database/` | PNG |
| **SIIM-ACR** | Kaggle | `siim/SIIM-ACR-Pneumothorax-original/` | DICOM / PNG |
| **RSNA Pneumonia** | Kaggle | `rsna/stage_2_train_images_png/`, `stage_2_test_images/`, `stage_2_train_labels.csv` | PNG |
| **RadGraph** | PhysioNet (credentialed) | `radgraph/train.json`, `dev.json`, `test.json` | JSON annotations |
| **RadNLI** | Public | `radnli/radnli_dev_v1.jsonl`, `radnli_test_v1.jsonl` | JSONL |
| **Rad-ReStruct** | Public | `Rad-ReStruct/train/`, `data/` | JSON + images |
| **Rex-Gradient-160K** | HuggingFace (gated) | `rex-gradient160k/deid_png/` | PNG (de-identified) |
| **VinDr-PCXR** | PhysioNet | `vindr-pcxr/` | PNG 512px |
| **MIMIC-CXR VQA** | PhysioNet (credentialed) | `mimiccxrvqa/`, `mimic-diff-vqa/`, `mimic-ext-mimic-cxr-vqa/` | JSON + MIMIC images |
| **MIMIC-NLE** | PhysioNet (credentialed) | `mimic-nle/` | JSON |
| **RadQA** | Public | `radqa/` | JSON |
| **CheXBench** | Public | `chexbench/metadata/` | JSON |
| **ReXVal** | Public | `rexval/` | JSON |

### 3. Preprocess images

After downloading, resize and normalize images for each dataset:

```bash
python src/preprocess/chexpert/preprocessImagesChexpert.py
python src/preprocess/brax/PreprocessBraxImages.py
python src/preprocess/chestxray14/preprocessImagesCxr14.py
python src/preprocess/openI/PreprocessOpenIImages.py
```

### 4. Set your HuggingFace token

The training pipeline reads credentials from the environment. **Never hardcode tokens in source files.**

```bash
export HF_TOKEN="hf_your_token_here"
```

Or add it to your SLURM script's environment section:

```bash
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set}"
```

---

## Fine-tuning

### Prerequisites

- Python 3.10+, CUDA 12.1+
- Install dependencies and activate the environment: `source activateEnv.sh`
- Submodules initialized: `git submodule update --init --recursive`
- CheXInstruct downloaded and images preprocessed (see above)
- HuggingFace token set: `export HF_TOKEN=...`

### Step 1 — Create the HuggingFace Parquet dataset

Convert the raw `.pkl` annotation files to the model's conversation format and save as HF Parquet:

```bash
python src/dataset/chexinstruct/createHFDataset.py \
    --model gemma \
    --input data_chexinstruct \
    --output data_chexinstruct/hf_parquet_gemma_format \
    --splits train val
```

Supported `--model` values: `gemma` · `llava15` · `qwen25vl`

This produces:

```
data_chexinstruct/hf_parquet_gemma_format/
└── gemma/
    ├── train/  (Parquet shards)
    └── val/    (Parquet shards)
```

### Step 2 — Tokenize the dataset

Pre-tokenize the formatted dataset to speed up training:

```bash
python src/dataset/chexinstruct/format_chexinstruct.py \
    --model_name_or_path google/gemma-3-4b-it \
    --dataset_dir data_chexinstruct/hf_parquet_gemma_format/gemma_findings \
    --output_dir gemma_3
```

Output will be saved to `data_chexinstruct/hf_parquet_gemma_format/gemma_3_findings_tok/`.

### Step 3 — Launch training

#### On Alvis (multi-node, recommended)

The main SLURM script targets 2 nodes × 4 A100 GPUs with DeepSpeed ZeRO-3:

```bash
sbatch accelerated_chexinstruct_gemma3_multinode_z3.sh
```

For A100-specific optimizations:

```bash
sbatch chexinstruct_gemma3_multinode_z3_a100.sh
```

For a debug/single-node run:

```bash
sbatch debug_chexinstruct_gemma3_onenode_z3.sh
```

#### Key training arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name_or_path` | `google/gemma-3-4b-it` | Base model |
| `--dataset_dir` | — | Path to tokenized Parquet dataset |
| `--output_dir` | — | Where to save checkpoints |
| `--learning_rate` | `2e-4` | Peak LR |
| `--lr_scheduler_type` | `cosine_with_restarts` | LR schedule |
| `--per_device_train_batch_size` | `4` | Batch size per GPU |
| `--gradient_accumulation_steps` | `4` | Effective batch = 128 (2 nodes × 4 GPUs × 4 × 4) |
| `--num_train_epochs` | `3` | Training epochs |
| `--lora_enable` | `true` | Enable LoRA |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `64` | LoRA alpha |
| `--peft_strategy` | `lora_gaussian` | LoRA weight initialization |
| `--bf16` | `true` | bfloat16 precision |
| `--gradient_checkpointing` | `true` | Memory-efficient backprop |
| `--model_max_length` | `1500` | Max token length |
| `--deepspeed_config_file` | `deepspeed/ds_zero3_config.yaml` | DeepSpeed config |

#### Monitoring

Training progress is logged to Weights & Biases. Set your project before launching:

```bash
export WANDB_PROJECT="chexinstruct-medgemma"
export WANDB_API_KEY="your_key"
```

#### Checkpoints

The script saves:
- **Per-epoch checkpoints**: `<output_dir>/epoch_N/`
- **Best checkpoint** (lowest eval perplexity): `<output_dir>/best_checkpoint/`
- **Final model**: `<output_dir>/` (safetensors format + tokenizer)
- **Training metadata**: `<output_dir>/all_results.json`

### Step 4 — Inference

Generate reports with a fine-tuned checkpoint:

```bash
python src/inference/generate_reports.py \
    --model_path reports/finetune_gemma_findings_zero3_trainer_lora64/best_checkpoint
```

---

## Data Pipeline

```
data_chexinstruct/data_train_chexinstruct.pkl   ← raw CheXInstruct annotations
         │
         ▼  createHFDataset.py  (format + serialize)
data_chexinstruct/hf_parquet_gemma_format/gemma/
         │
         ▼  format_chexinstruct.py  (tokenize)
data_chexinstruct/hf_parquet_gemma_format/gemma_3_findings_tok/
         │
         ▼  finetune_accelerated_v2.py  (train)
reports/finetune_gemma_findings_zero3_trainer_lora64/
```

---

## Evaluation

Evaluation uses a suite of radiology-specific metrics via the submodules:

- **GREEN score** (`modules/GREEN`) — Clinically-grounded report evaluation
- **ReXrank** (`modules/ReXrank`) — Ranking-based radiology report assessment
- **cxr-metrics** (`modules/cxr-metrics`) — CXR NLP metrics (BLEU, ROUGE, RadGraph F1, etc.)

Inference for evaluation:

```bash
python src/inference/generate_reports.py
python src/benchmark/benchmark_VLM.py
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU(s)
- HuggingFace account with access to `google/gemma-3-4b-it`

### Environment

```bash
source activateEnv.sh
```

### Submodules

```bash
git submodule update --init --recursive
```

### HuggingFace token

Set your token in the environment or in the training scripts:

```bash
export HF_TOKEN="your_token_here"
```

---

## Citation

If you use this framework or the CheXInstruct dataset, please cite the original CheXInstruct paper and the relevant source datasets.

---

## Acknowledgements

This work was conducted on the **Alvis HPC cluster** provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), under project NAISS2024-5-577.
