# Applied Deep Learning (ADL) - NTU 2023 Fall

This repository contains the homework assignments and final project for the Applied Deep Learning (ADL) course at National Taiwan University (NTU) in the Fall of 2023.

## Repository Structure

The repository is organized into four main directories:

- **hw1/**
- **hw2/**
- **hw3/**
- **Final_Project/**

### Directory Breakdown

#### 1. [hw1](./hw1)
**Task**: Paragraph and Span Selection

This assignment focuses on implementing two distinct models: 
- **Paragraph Selection Model**: Given a Chinese question and four Chinese paragraphs, the model identifies the paragraph most relevant to the question.
- **Span Selection Model**: After selecting the relevant paragraph, this model extracts the precise answer span from the chosen paragraph.

Key Concepts: Text classification, Span extraction, Chinese Natural Language Processing (NLP).

#### 2. [hw2](./hw2)
**Task**: Fine-tuning mT5 for Title Generation

In this assignment, we fine-tune a pre-trained Multilingual Text-to-Text Transfer Transformer (mT5) to generate appropriate titles for given paragraphs. The model performance is evaluated using the Rouge score, a standard metric for sequence generation tasks.

Key Concepts: mT5, Sequence generation, Rouge score, NLP in multiple languages.

#### 3. [hw3](./hw3)
**Task**: Instruction-tuning Taiwan LlaMa Model

This assignment involves instruction-tuning a Taiwan LlaMa Model for a language translation task. The model is trained to either:
- Translate plain Chinese into classical Chinese, or
- Translate classical Chinese into plain Chinese.

Performance is evaluated using the **Perplexity** metric.

Key Concepts: Instruction-tuning, Language translation, Classical Chinese NLP.


