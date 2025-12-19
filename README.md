# ETSP — Multilingual Counterspeech Generation

This repository contains the code for a **semester project** on **Multilingual Counterspeech Generation**, conducted as part of a university course and inspired by the **Shared Task on Multilingual Counterspeech Generation (MCG-COLING 2025)**.

The project investigates how **large language models fine-tuned with parameter-efficient methods** perform across **high- and low-resource languages**, with a focus on multilingual generalization and training dynamics.

---

## Project Overview

- **Task**: Generate respectful and factual counter-narratives in response to hate speech  
- **Languages**: English, Italian, Spanish, Basque  
- **Model**: LLaMA-2-7B  
- **Fine-tuning**: LoRA + 4-bit quantization (NF4)  
- **Evaluation**: BLEU, ROUGE-L, BERTScore  

---

## Repository Structure

- **Main training and evaluation notebook**: Counterspeech.py
- **Knowledge selection**: kwnoledge_selection_gpt4o.py
-  **Dataset used for knowledge selection part**: train_selected.csv, validation_selected.csv
- **Project documentation**: README.md 


## Dataset

This project uses the dataset released for the **MCG-COLING Shared Task**, consisting of:

- **596 Hate Speech – Counter Narrative pairs per language**
- **5 background knowledge sentences per instance**
- **Splits**:
  - Train: 396
  - Development: 100
  - Test: 100

Dataset source:
- Hugging Face: `LanD-FBK/ML_MTCONAN_KN`

---

## Methodology

### Model
- **Base model**: `meta-llama/Llama-2-7b-hf`
- **Objective**: Causal Language Modeling

### Fine-Tuning
- **LoRA rank**: 8  
- **LoRA alpha**: 64  
- **Dropout**: 0.05  
- **Quantization**: 4-bit NF4 using `bitsandbytes`  
- **Frameworks**: Hugging Face Transformers, PEFT

Each language is fine-tuned **independently** using identical hyperparameters to enable fair cross-lingual comparison.

---

## Evaluation

Model performance is evaluated using the following automatic metrics:

- **BLEU**
- **ROUGE-L**
- **BERTScore**

Training dynamics are monitored using **TensorBoard** to analyze convergence behavior across languages.

---

## How to Run

### 1. Install dependencies

```bash
pip install torch transformers datasets peft bitsandbytes accelerate evaluate bert_score rouge_score
```

### 2. Run training and evaluation

Open and run the Python file:

```bash
Counterspeech.py
```

A GPU with sufficient VRAM is strongly recommended.
