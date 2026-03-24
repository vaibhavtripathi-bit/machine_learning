# Machine Learning Projects Portfolio

A comprehensive collection of ML projects progressing from classical machine learning to production systems. Built as part of a structured learning roadmap.

## Progress Tracker

| # | Project | Tech Stack |
|---|---------|--------|------------|----------|
| 01 | [Spam Email Classifier](./01_spam_email_classifier/) |  scikit-learn, pandas|
| 02 | [House Price Predictor](./02_house_price_predictor/) | XGBoost, pandas, matplotlib  |
| 03 | [Image Classifier](./03_image_classifier/) | PyTorch, torchvision, ResNet |
| 04 | [Sentiment Analyzer](./04_sentiment_analyzer/) | PyTorch, HuggingFace, BERT  |
| 05 | [RAG Q&A Bot](./05_rag_qa_bot/) | LangChain, ChromaDB, OpenAI  |
| 06 | [Fine-tuned LLM](./06_finetuned_llm/) | HuggingFace, PEFT, LoRA  |
| 07 | [Android ML App](./07_android_ml_app/) | TFLite, ONNX, Kotlin |
| 08 | [Production ML System](./08_production_ml_system/) | MLflow, FastAPI, Docker  |

## Project Categories

### Classical ML (Projects 01-02)
Foundation projects covering the complete ML pipeline: data preprocessing, feature engineering, model training, and evaluation.

### Deep Learning (Projects 03-04)
Introduction to neural networks, CNNs for computer vision, and transformers for NLP.

### LLMs & GenAI (Projects 05-06)
Modern AI applications including RAG pipelines and LLM fine-tuning with PEFT/LoRA.

### Mobile & Production (Projects 07-08)
On-device ML deployment and production-grade ML systems with monitoring.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/vaibhavtripathi-bit/machine_learning.git
cd machine_learning

# Navigate to any project
cd 01_spam_email_classifier

# Install dependencies
pip install -r requirements.txt

# Run the project
python src/main.py
```

## Tech Stack Overview

| Category | Technologies |
|----------|-------------|
| **Classical ML** | scikit-learn, XGBoost, pandas, numpy |
| **Deep Learning** | PyTorch, torchvision, HuggingFace Transformers |
| **LLMs** | LangChain, OpenAI API, ChromaDB, PEFT |
| **Mobile ML** | TensorFlow Lite, ONNX Runtime, Kotlin |
| **Production** | MLflow, FastAPI, Docker, GitHub Actions |
| **Visualization** | matplotlib, seaborn, Gradio, Streamlit |

## Repository Structure

```
machine_learning/
├── 01_spam_email_classifier/    # Binary text classification
├── 02_house_price_predictor/    # Regression with feature engineering
├── 03_image_classifier/         # CNN + transfer learning
├── 04_sentiment_analyzer/       # LSTM vs BERT comparison
├── 05_rag_qa_bot/               # Retrieval-augmented generation
├── 06_finetuned_llm/            # LoRA fine-tuning
├── 07_android_ml_app/           # On-device inference
├── 08_production_ml_system/     # End-to-end MLOps
└── shared/                      # Common utilities
```

## Learning Resources

- [fast.ai](https://www.fast.ai/) - Deep learning courses
- [HuggingFace Course](https://huggingface.co/course) - Transformers & NLP
- [Made With ML](https://madewithml.com/) - MLOps best practices

## License

MIT License
