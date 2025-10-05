import os
import argparse
import pandas as pd
from source.model_preparation import prepare_X_y, split
from source.data_preprocessing import preprocessing_data
from source.model_building_bert import build_pipeline


def main(args=None):
    parser = argparse.ArgumentParser(description='Run training pipeline (BERT/PhoBERT etc.)')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='HF model id')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--save_dir', type=str, default='outputs')
    parser.add_argument('--dry_run', action='store_true', help='Do a quick forward-pass instead of full training')
    parser.add_argument('--quick', action='store_true', help='Use a very small model for quick testing')
    parsed = parser.parse_args(args=args)

    # Data Preprocessing
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    PATH1 = os.path.join(repo_root, 'datasets', 'DataSet_Misinfo_TRUE.csv')
    PATH2 = os.path.join(repo_root, 'datasets', 'DataSet_Misinfo_FAKE.csv')

    if not os.path.exists(PATH1) or not os.path.exists(PATH2):
        print(f"Data files not found at paths: {PATH1}, {PATH2}")
        print("Put the CSV files in the current working directory or update the paths in source/main.")
        return

    df_true = pd.read_csv(PATH1)
    df_true['target'] = 1

    df_fake = pd.read_csv(PATH2)
    df_fake['target'] = 0

    df = pd.concat([df_true, df_fake], ignore_index=True)

    # Model Preparation
    df = preprocessing_data(df)
    X, y = prepare_X_y(df)
    trainX, testX, trainY, testY = split(X, y, train_size=0.8, random_state=42)

    # Choose a very small model for quick testing if requested
    model_name = parsed.model_name
    if parsed.quick:
        print("Quick mode: switching to a tiny model for fast execution")
        model_name = 'sshleifer/tiny-distilbert-base-uncased'

    # Build pipeline
    pipeline = build_pipeline(
        train_texts=trainX['text'].tolist(),
        train_labels=trainY.tolist(),
        test_texts=testX['text'].tolist(),
        test_labels=testY.tolist(),
        model_name=model_name,
        num_labels=2,
        max_length=parsed.max_length,
        batch_size=parsed.batch_size,
        from_pt=False,
        lr=parsed.lr,
        epochs=parsed.epochs
    )

    tokenizer = pipeline['tokenizer']
    model = pipeline['model']
    trainer = pipeline['trainer']

    if parsed.dry_run:
        print('Dry run: performing a single forward pass on one batch')
        # take one batch
        batch = next(iter(pipeline['train_loader']))
        device = trainer.device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print('Logits shape:', outputs.logits.shape)
        return

    # Train full model
    trainer.train()

    # Evaluate and save
    results = trainer.evaluate() if pipeline['test_loader'] is not None else {}

    save_dir = parsed.save_dir
    os.makedirs(save_dir, exist_ok=True)
    safe_name = model_name.replace('/', '_')
    model_out = os.path.join(save_dir, safe_name)
    os.makedirs(model_out, exist_ok=True)
    print(f"Saving model and tokenizer to {model_out}")
    try:
        model.save_pretrained(model_out)
        tokenizer.save_pretrained(model_out)
    except Exception as e:
        print('Failed to save model/tokenizer:', e)
    print('Done')
