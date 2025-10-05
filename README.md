# Fake-News-Detection-Project

Purpose: A sample project for training a fake-news detection classifier using Transformer models (BERT, PhoBERT, etc.).

## What this document covers
- Short introduction
- Project layout
- Requirements & installation (Windows PowerShell)
- How to run (preprocessing, training, evaluation)
- Notes about the dataset

## Main project structure

```
Fake-News-Detection-Project/
├─ README.md                         # (this file)
├─ requirements.txt                   # Python dependencies
├─ datasets/                          # Folder for dataset CSVs
│  ├─ DataSet_Misinfo_FAKE_trimmed.csv
│  └─ DataSet_Misinfo_TRUE_trimmed.csv
├─ source/                            # Main source code
│  ├─ __init__.py
│  ├─ data_preprocessing.py           # Text cleaning and preprocessing
│  ├─ main.py                         # Entry point: preprocessing + training
│  ├─ model_building.py               # (build pipeline / model)
│  ├─ model_preparation.py            # Prepare X/y and train/test split
│  └─ model_building_bert.py          # Example pipeline using transformers (imported by main)
├─ tools/
│  ├─ trim_datasets.py                # Utility to trim CSV files by size
│  └─ trim_datasets.ps1               # PowerShell wrapper script
└─ other files...
```


## Requirements

- Python 3.8+ (recommended 3.8 - 3.11)
- GPU recommended for faster training (optional)
- Python packages listed in `requirements.txt` (e.g. transformers, torch, datasets, scikit-learn, pandas, tqdm, regex)

## Installation (Windows PowerShell)

1. Open PowerShell in the project folder:

```powershell
cd D:\Repos\Fake-News-Detection-Project
```

2. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run this once (follow your organization policies):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

3. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Note: `requirements.txt` includes packages such as `transformers`, `torch`, `datasets`, `scikit-learn`, `pandas`, `tqdm`, `regex`. For `torch` you may want to install the version matching your CUDA setup — see https://pytorch.org/ for the recommended install command.

## Preparing the dataset

1. Place your CSV files inside the `datasets/` folder.
	- If the original CSVs are large, use `tools/trim_datasets.py` or `tools/trim_datasets.ps1` to create smaller `_trimmed.csv` versions.

2. Example: use the Python trim utility to limit file size (keeps header and writes rows until the size limit):

```powershell
python tools\trim_datasets.py datasets\DataSet_Misinfo_TRUE.csv --max_mb 50
python tools\trim_datasets.py datasets\DataSet_Misinfo_FAKE.csv --max_mb 50
```

Or use the PowerShell wrapper if available:

```powershell
.\tools\trim_datasets.ps1 datasets\DataSet_Misinfo_TRUE.csv datasets\DataSet_Misinfo_FAKE.csv -max_mb 50
```

3. After trimming you will have files like `DataSet_Misinfo_TRUE_trimmed.csv` and `DataSet_Misinfo_FAKE_trimmed.csv`. Either rename them or update `source/main.py` to use those names.

## Running training (example)

Run the training pipeline with default options (PowerShell):

```powershell
python -m source.main --model_name bert-base-uncased --epochs 3 --batch_size 16 --max_length 256 --save_dir outputs
```

Quick test mode (fast sanity check):

```powershell
python -m source.main --quick --dry_run
```

`--quick` uses a tiny model for fast execution. `--dry_run` performs a single forward pass to verify the pipeline.

## Utility scripts

- `tools/trim_datasets.py` — trim a CSV file by size (bytes/MB). Useful to create `_trimmed.csv` copies for quick experiments.

## Saving the model

After training, `source/main.py` saves the model and tokenizer to the directory set by `--save_dir`. A subfolder is created using the `model_name` (slashes `/` are replaced with `_`).
