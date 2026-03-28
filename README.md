# Fake News Detection Web App

A complete NLP + TensorFlow fake news detection application with a premium glassmorphism web interface. The app predicts whether input news text is REAL or FAKE and returns a confidence score with practical guidance.

## Features

- FastAPI backend with robust validation and JSON prediction API
- TensorFlow BiLSTM text classifier with dropout and class balancing
- Shared preprocessing pipeline for training and inference
- Tokenization + sequence padding workflow
- Confidence-aware guidance messaging
- Premium glassmorphism frontend with animated background and responsive layout
- Loading states, error handling, confidence progress bar, and prediction badge
- Local history of recent checks in browser storage
- Sample CSV dataset format included

## Project Structure

- app.py
- train_model.py
- utils/text_processing.py
- templates/index.html
- static/style.css
- static/script.js
- dataset/news.csv
- model/
- requirements.txt
- README.md

## API Contract

### POST /predict

Request JSON:

```json
{
	"text": "Your news content goes here"
}
```

Success response:

```json
{
	"prediction": "REAL",
	"confidence": 0.91,
	"status": "success",
	"guidance": "Likely real. Continue verifying with additional trusted sources."
}
```

Error responses include validation details for empty, short, or invalid input.

## Dataset Format

CSV file must include one text column and one label column.

Accepted text column names:
- text
- news
- content
- article
- statement

Accepted label column names:
- label
- class
- target
- is_fake

Accepted label values:
- REAL, TRUE, 1, GENUINE
- FAKE, FALSE, 0, FRAUD

Example:

```csv
text,label
"Government confirms new public health initiative after regulatory review.",REAL
"A viral post claims moonlight cures all diseases instantly.",FAKE
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py --data dataset/news.csv --model-dir model
```

If you are using uploaded Kaggle files in the dataset folder, train with:

```bash
python train_model.py --fake-data dataset/Fake.csv --real-data dataset/True.csv --model-dir model
```

For faster training, use a larger batch or full-batch mode:

```bash
python train_model.py --fake-data dataset/Fake.csv --real-data dataset/True.csv --model-dir model --batch-size 512
```

```bash
python train_model.py --fake-data dataset/Fake.csv --real-data dataset/True.csv --model-dir model --full-batch
```

If your machine is slow with LSTM recurrent dropout, keep it at the default (`--recurrent-dropout 0.0`).

This generates:
- model/fake_news_model.keras
- model/tokenizer.json
- model/metadata.json
- model/metrics.json

## Run the App

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- http://localhost:8000

## Notes for Best Accuracy

- Use a larger real-world dataset for production accuracy improvements.
- Keep labels balanced between REAL and FAKE classes.
- Provide complete article-level text instead of short headlines.