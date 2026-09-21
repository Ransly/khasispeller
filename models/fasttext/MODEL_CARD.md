# Khasi FastText Word Embedding Model

## Overview
This repository contains a subword-based word embedding model trained on Khasi text using Gensim FastText.

The model supports out-of-vocabulary (OOV) words through character n-gram representations, making it suitable for morphologically rich and low-resource languages like Khasi.

---

## Model Details

- Model Type: FastText (subword embeddings)
- Library: Gensim
- Vector Size: 100
- Training: Unsupervised

---

## Files

- `word2vec_model.model`  
  → Full Gensim FastText model (required for usage)

- `word2vec_model.model.wv.vectors_ngrams.npy`  
  → Internal character n-gram embedding matrix (shape: 2,000,000 × 100)

---

## Important Notes

- The `.model` file is required for inference.
- The `.npy` file alone is NOT usable:
  - It contains hashed n-gram vectors
  - It does not store vocabulary or mappings

---

## Usage

```python
from gensim.models import FastText

model = FastText.load("word2vec_model.model")

# Get vector for a word (works even if word is unseen)
vector = model.wv["jingbha"]

# Similarity
sim = model.wv.similarity("jingbha", "bha")
