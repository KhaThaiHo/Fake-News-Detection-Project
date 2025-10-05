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

## Project Introduction
### Problem Statement

The widespread dissemination of fake news and various forms of misinformation has caused serious risks to society, including eroding public trust, increasing political polarization, manipulating elections, and posing particular dangers during pandemics or conflicts. From a Natural Language Processing (NLP) research perspective, identifying fake news remains a challenging task.

Linguistically, fake news often imitates the style and structure of legitimate journalism, making formal features less effective for distinguishing between truth and falsehood. The lack of reliable and up-to-date labeled datasets, especially across different languages and regions, has reduced the effectiveness of supervised learning models. The adaptability and resilience of fake news allow malicious actors to constantly change language and tactics to evade detection systems. Additionally, context, culture, attitudes, and implicit biases contribute to the complexity of analysis.

Moreover, NLP models may amplify training data bias, leading to unfair classifications and censorship of legitimate content. These challenges emphasize the need for cautious, context-aware approaches; if not handled carefully, such systems might unintentionally contribute to the spread of misinformation.

### Dataset
The dataset consists of two parts:

- MisinfoSuperset_TRUE.csv: A collection of verified and legitimate news articles published by reputable sources such as Reuters, The New York Times, The Washington Post, and others.

- MisinfoSuperset_FAKE.csv: Contains sophisticated fake news excerpts sourced from American right-wing extremist websites (e.g., Redflag Newsdesk, Breitbart, Truth Broadcast Network), and from
Ahmed, H., Traore, I., & Saad, S. (2017): “Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques” (Springer LNCS 10618).

## Implementation Process
### Data Preprocessing and Analysis

The process includes the following steps:

- Cleaning the data.

- Visualizing the Top 10 most frequent words in each dataset.

### Model Training
#### Traditional Models

Approaches used:

- TF-IDF Vectorizer

- Logistic Regression

- Support Vector Machine (SVM)

#### Transformer-Based Models

For each Transformer model, the following steps are executed sequentially:

- Tokenization

- Data Loader construction

- Model Building

- Parameter Tuning

- Model Training

- Performance Evaluation

- Inference

Defined classes:

- TextClassificationDataset and DataLoaderBuilder: Standardize and tokenize input data.

- Trainer: Includes the train() function for training, evaluate() for performance evaluation, and predict() for inference on new input.

Models used:

- BERT

- XLNet

- RoBERTa

## Conclusion

In general, both traditional and deep learning-based models can effectively train and predict on the dataset, aiding humans in better recognizing and combating fake news. Models such as BERT, XLNet, and RoBERTa perform almost perfectly on the provided English-language dataset, as they were pre-trained on large English corpora.

However, several challenges remain:

- High computational cost: Although academic platforms like Google Colab and Kaggle provide GPU resources for training, they impose time limits. Otherwise, extensive resources and funding are required for long-term experiments.

- Outdated datasets: Since the dataset may contain old information, real-world predictions might be inaccurate if the model relies on outdated or invalid data.

- Limited domain diversity: The dataset may focus only on specific domains (Politics, Economics, News, etc.) or sources, causing stylistic homogeneity.

- Language constraint: The dataset is in English — an advantage for pre-trained Transformer models — but performance may degrade if applied to Vietnamese or other languages.

Future Work and Recommendations

- Expand and update the dataset regularly using multiple credible news sources to capture diverse journalistic styles.

- Develop and incorporate multilingual datasets.

- Explore additional models such as PhoBERT for Vietnamese text.

- Fine-tune hyperparameters for improved performance.