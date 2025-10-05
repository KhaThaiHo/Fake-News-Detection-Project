"""Reusable model-building utilities for sequence classification.

This module provides generic helpers to load any Hugging Face model/tokenizer
for sequence classification (BERT, PhoBERT, XLNet, etc.), build dataloaders,
and train/evaluate a model with a small Trainer wrapper.

Primary convenience API:
	build_pipeline(train_texts, train_labels, test_texts=None, test_labels=None, model_name=..., ...)

The functions are intentionally dependency-light and accept plain Python lists / pandas Series
for texts and labels so they can be called from notebooks or scripts.
"""

from typing import List, Optional, Tuple, Dict, Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import torch.nn as nn
from datasets import load_dataset
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def load_model_tokenizer(model_name: str, num_labels: int = 2, from_pt: bool = False):
	"""Load a tokenizer and a sequence-classification model from Hugging Face.

	Args:
		model_name: model id (e.g. 'bert-base-uncased', 'vinai/phobert-base')
		num_labels: number of labels for the classification head
		from_pt: if True, treat provided weights as PyTorch weights; if False, call from_pretrained with from_tf when needed

	Returns:
		tokenizer, model
	"""
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	# from_pt toggling isn't always necessary; using from_pretrained default should work for most models
	model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, from_tf=not from_pt)
	return tokenizer, model


class TextClassificationDataset(Dataset):
	def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
		self.texts = list(texts)
		self.labels = list(labels) if labels is not None else None
		self.tokenizer = tokenizer
		self.max_length = max_length

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		encoding = self.tokenizer(
			self.texts[idx],
			truncation=True,
			padding='max_length',
			max_length=self.max_length,
			return_tensors='pt'
		)

		item = {
			'input_ids': encoding['input_ids'].squeeze(0),
			'attention_mask': encoding['attention_mask'].squeeze(0),
		}
		if self.labels is not None:
			item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
		return item


class DataLoaderBuilder:
	def __init__(self, tokenizer, max_length: int = 256, batch_size: int = 16, shuffle: bool = True):
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.batch_size = batch_size
		self.shuffle = shuffle

	def get_dataloader(self, texts, labels=None) -> DataLoader:
		dataset = TextClassificationDataset(texts, labels, self.tokenizer, self.max_length)
		return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)


class Trainer:
	def __init__(self, model, train_loader: DataLoader, test_loader: Optional[DataLoader] = None, tokenizer=None, max_length: int = 256, lr: float = 2e-5, epochs: int = 3, device: Optional[str] = None):
		self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = model.to(self.device)
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.tokenizer = tokenizer
		self.max_length = max_length
		self.epochs = epochs

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = AdamW(self.model.parameters(), lr=lr)
		# scheduler length requires knowing steps; ensure train_loader is not empty
		total_steps = max(1, len(train_loader) * max(1, epochs))
		self.scheduler = get_scheduler(
			"linear",
			optimizer=self.optimizer,
			num_warmup_steps=0,
			num_training_steps=total_steps
		)

	def train(self, show_progress: bool = True):
		self.model.train()
		for epoch in range(self.epochs):
			total_loss = 0.0
			iterator = self.train_loader
			if show_progress:
				iterator = tqdm(iterator, desc=f"Epoch {epoch+1}/{self.epochs}")

			for batch in iterator:
				input_ids = batch['input_ids'].to(self.device)
				attention_mask = batch['attention_mask'].to(self.device)
				labels = batch['labels'].long().to(self.device)

				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
				logits = outputs.logits

				loss = self.criterion(logits, labels)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				try:
					self.scheduler.step()
				except Exception:
					pass

				total_loss += loss.item()
				if show_progress:
					iterator.set_postfix(loss=loss.item())

			avg_loss = total_loss / max(1, len(self.train_loader))
			print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

			if self.test_loader is not None:
				self.evaluate()

	def evaluate(self) -> Dict[str, Any]:
		self.model.eval()
		total_loss = 0.0
		preds_all = []
		labels_all = []

		with torch.no_grad():
			for batch in self.test_loader:
				input_ids = batch['input_ids'].to(self.device)
				attention_mask = batch['attention_mask'].to(self.device)
				labels = batch['labels'].long().to(self.device)

				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
				logits = outputs.logits

				loss = self.criterion(logits, labels)
				total_loss += loss.item()

				preds = torch.argmax(logits, dim=1)
				preds_all.extend(preds.cpu().numpy())
				labels_all.extend(labels.cpu().numpy())

		avg_loss = total_loss / max(1, len(self.test_loader))
		acc = accuracy_score(labels_all, preds_all) if labels_all else 0.0
		print(f"Test Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")
		report = classification_report(labels_all, preds_all, zero_division=0)
		print(report)
		self.model.train()
		return {
			'loss': avg_loss,
			'accuracy': acc,
			'classification_report': report,
			'preds': preds_all,
			'labels': labels_all
		}

	def predict(self, texts: List[str]) -> List[int]:
		self.model.eval()
		class PredictionDataset(Dataset):
			def __init__(self, texts, tokenizer, max_length):
				self.texts = list(texts)
				self.tokenizer = tokenizer
				self.max_length = max_length

			def __len__(self):
				return len(self.texts)

			def __getitem__(self, idx):
				encoding = self.tokenizer(
					self.texts[idx],
					truncation=True,
					padding='max_length',
					max_length=self.max_length,
					return_tensors='pt'
				)
				return {
					'input_ids': encoding['input_ids'].squeeze(0),
					'attention_mask': encoding['attention_mask'].squeeze(0),
				}

		prediction_dataset = PredictionDataset(texts, self.tokenizer, self.max_length)
		prediction_loader = DataLoader(prediction_dataset, batch_size=self.train_loader.batch_size, shuffle=False)

		all_preds = []
		with torch.no_grad():
			for batch in tqdm(prediction_loader, desc="Predicting"):
				input_ids = batch['input_ids'].to(self.device)
				attention_mask = batch['attention_mask'].to(self.device)

				outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
				logits = outputs.logits

				if getattr(self.model.config, 'num_labels', 2) == 1:
					probs = torch.sigmoid(logits).squeeze()
					preds = (probs > 0.5).long()
				else:
					preds = torch.argmax(logits, dim=1)
				all_preds.extend(preds.cpu().numpy())

		return all_preds


def build_pipeline(train_texts: List[str], train_labels: List[int], test_texts: Optional[List[str]] = None, test_labels: Optional[List[int]] = None, model_name: str = 'bert-base-uncased', num_labels: int = 2, max_length: int = 256, batch_size: int = 16, shuffle: bool = True, from_pt: bool = False, lr: float = 2e-5, epochs: int = 3) -> Dict[str, Any]:
	"""High-level helper to build tokenizer, model, dataloaders and trainer.

	Returns a dict with keys: tokenizer, model, train_loader, test_loader (may be None), trainer
	"""
	tokenizer, model = load_model_tokenizer(model_name, num_labels=num_labels, from_pt=from_pt)

	loader_builder = DataLoaderBuilder(tokenizer, max_length=max_length, batch_size=batch_size, shuffle=shuffle)
	train_loader = loader_builder.get_dataloader(train_texts, train_labels)
	test_loader = None
	if test_texts is not None and test_labels is not None:
		test_loader = loader_builder.get_dataloader(test_texts, test_labels)

	trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader, tokenizer=tokenizer, max_length=max_length, lr=lr, epochs=epochs)

	return {
		'tokenizer': tokenizer,
		'model': model,
		'train_loader': train_loader,
		'test_loader': test_loader,
		'trainer': trainer,
	}


__all__ = [
	'load_model_tokenizer',
	'TextClassificationDataset',
	'DataLoaderBuilder',
	'Trainer',
	'build_pipeline',
]