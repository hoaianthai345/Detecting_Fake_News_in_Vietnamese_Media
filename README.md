# Vietnamese Fake News Detection: A Comprehensive Evaluation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.36.2-orange.svg)](https://huggingface.co/transformers/)

A comprehensive comparative study of machine learning approaches for Vietnamese fake news detection, comparing traditional deep learning, transfer learning, and large language models.

## 🏆 Key Results

- **Best Performance**: PhoBERT fine-tuned achieves **96.30% accuracy**
- **Performance Gap**: 23-percentage-point improvement over best LLM
- **Efficiency**: PhoBERT offers optimal accuracy-resource balance
- **Class Imbalance**: Successfully handles severe imbalance (83.2% vs 16.8%)

## 📊 Model Performance Summary

| Model Family | Best Model | Accuracy | F1-Score | Inference Time |
|--------------|------------|----------|----------|----------------|
| **Transfer Learning** | PhoBERT (fine-tuned) | **96.30%** | **88.52%** | ~20ms |
| Traditional DL | BiLSTM + FastText | 81.89% | 44.83% | ~15ms |
| Large Language Models | Qwen2.5 (zero-shot) | 73.25% | 42.15% | ~3s |

## 🗂️ Repository Structure
vietnamese-fake-news-detection/
├── BiLSTM/
│ ├── BiLSTM.ipynb # BiLSTM with random embeddings
│ ├── BiLSTM_word2vec.ipynb # BiLSTM with Word2Vec
│ └── BiLSTM_fasttext.ipynb # BiLSTM with FastText
├── PhoBERT/
│ ├── PHOBERT.ipynb # PhoBERT fine-tuned (best model)
│ └── PHOBERT_pretrain.ipynb # PhoBERT frozen
├── LLMs/
│ ├── Qwen25_zeroshot.ipynb # Qwen2.5 zero-shot
│ ├── Qwen25_fewshot.ipynb # Qwen2.5 few-shot
│ ├── Llama2_zeroshot.ipynb # Llama-2 zero-shot
│ ├── Llama2_fewshot.ipynb # Llama-2 few-shot
│ └── deepseek.ipynb # DeepSeek evaluation
└── README.md # This file


## 🚀 Quick Start

### 1. Environment Setup

**Option A: Google Colab (Recommended)**
```python
# Simply open any notebook in Google Colab
# All dependencies will be installed automatically
```

**Option B: Local Setup**
```bash
git clone https://github.com/[username]/vietnamese-fake-news-detection.git
cd vietnamese-fake-news-detection
pip install -r requirements.txt
```

### 2. Dataset

Download the ReINTEL dataset from [VLSP 2020](https://vlsp.org.vn/vlsp2020/eval/reintel):
- **Total**: 9,713 Vietnamese social media posts
- **Classes**: Real news (83.2%) vs Fake news (16.8%)
- **Split**: 80% train, 10% validation, 10% test

### 3. Run Experiments

**Best Model (PhoBERT Fine-tuned):**
```bash
# Open PhoBERT/PHOBERT.ipynb in Colab or Jupyter
# Run all cells to reproduce 96.30% accuracy
```

**Compare All Models:**
```bash
# Run notebooks in each folder:
# - BiLSTM/ for traditional deep learning
# - PhoBERT/ for transfer learning  
# - LLMs/ for large language models
```

## 🎯 Model Families

### 1. Traditional Deep Learning
- **BiLSTM** with three embedding strategies:
  - Random initialization
  - Vietnamese Word2Vec ([PhoW2V](https://github.com/VinAIResearch/PhoW2V))
  - Vietnamese FastText

### 2. Transfer Learning
- **PhoBERT** ([VinAI Research](https://huggingface.co/vinai/phobert-base))
  - Frozen configuration
  - **Fine-tuned configuration** (best performance)

### 3. Large Language Models
- **Qwen2.5-7B** (zero-shot & few-shot)
- **Llama-2-7B Vietnamese** (zero-shot & few-shot)
- **DeepSeek** (Vietnamese legal domain)

## 📈 Key Findings

### Performance Insights
1. **Language-specific pre-training** significantly outperforms multilingual models
2. **Fine-tuning** beats prompt-based learning for specialized tasks
3. **Few-shot learning** sometimes degrades performance vs zero-shot
4. **Class imbalance** requires sophisticated model architectures

### Efficiency Analysis
- **PhoBERT**: Best accuracy-efficiency trade-off (2GB GPU, 20ms inference)
- **BiLSTM**: Fastest but poor minority class detection
- **LLMs**: High resource requirements (8GB GPU, 3+ second inference)

## 🛠️ Technical Details

### Hardware Requirements
- **Minimum**: 2GB GPU memory (PhoBERT)
- **Recommended**: 8GB GPU memory (for LLMs)
- **Platform**: Google Colab Pro or equivalent

### Key Dependencies
```python
torch==2.1.0
transformers==4.36.2
datasets==2.17.0
scikit-learn==1.4.1
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### Model Sources
- **PhoBERT**: `vinai/phobert-base`
- **Qwen2.5**: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- **Llama-2**: `ngoan/Llama-2-7b-vietnamese-20k`
- **DeepSeek**: `vulong3896/vnlegalqa-DeepSeek-R1-0528-Qwen3-8B-finetuned`

## 📊 Reproduce Results

### Performance Metrics
```python
# All results in results/performance_metrics.csv
PhoBERT (fine-tuned): 96.30% accuracy, 88.52% F1
BiLSTM + FastText:    81.89% accuracy, 44.83% F1
Qwen2.5 (zero-shot): 73.25% accuracy, 42.15% F1
```

## 🔬 Research Context

This research was conducted for **BGRA 2025 Competition** and provides:

- **First comprehensive benchmark** for Vietnamese fake news detection
- **Systematic comparison** across three major ML paradigms
- **Practical deployment guidelines** for Vietnamese NLP applications
- **Evidence** for language-specific model importance

## 🙏 Acknowledgments

- **VinAI Research** for PhoBERT model
- **VLSP 2020** for ReINTEL dataset
- **Hugging Face** for model hosting and transformers library
- **Google Colab** for computational resources

---

⭐ **Star this repository** if you find it helpful for your Vietnamese NLP research!
