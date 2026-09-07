# ML Journey 🤖

A self-directed machine learning learning path from Python fundamentals 
to deployed AI systems — built project by project, concept by concept.

## 🚀 Live Demo
**Titanic Survival Prediction API** — deployed and publicly accessible:

POST https://titanic-api-85fj.onrender.com/predict

Example request:
```json
{"Pclass": 1, "Age": 25, "Fare": 100, "Sex": 1}
```
Example response:
```json
{"survived": 1}
```

## 📁 Projects

### Stage 0 — Data Analysis (Titanic Dataset)
- Loaded and cleaned real-world data using Pandas
- Handled missing values (filled Age/Embarked, dropped Cabin)
- Visualized survival rates by passenger class using Matplotlib
- **Key insight**: 38.4% survival rate overall; 1st class passengers 
  had significantly higher survival rates

### Stage 2 — Core Machine Learning
**Titanic Survival Classification** (`titanic.py`)
- Model: Logistic Regression (scikit-learn)
- Features: Pclass, Age, Fare, Sex
- **Accuracy: 80.4%** on held-out test set
- Baseline comparison: beats random guessing (50%) and 
  naive "always predict no" baseline (62%)

**California House Price Regression** (`house_prices.py`)
- Model: Linear Regression (scikit-learn)
- Improved MAE from $60,330 (3 features) to $53,320 (all 8 features)
  by adding location, occupancy, and population data
- **Key lesson**: more relevant features → lower prediction error

**Neighborhood Segmentation** (`house_prices.py`)
- Model: K-Means Clustering (scikit-learn)
- Discovered 3 distinct neighborhood types based on income and house age
- **Key lesson**: feature scaling (StandardScaler) is critical for 
  distance-based algorithms — unscaled data produced misleading clusters

### Stage 3 — Deep Learning (MNIST Handwritten Digits)
**Simple Neural Network** (`mnist.py`)
- Built from scratch using PyTorch
- Architecture: Flatten → Linear(784,128) → ReLU → Linear(128,10)
- **Accuracy: 96.82%** on 10,000 test images

**Convolutional Neural Network (CNN)**
- Added Conv2d + MaxPool2d layers to preserve spatial structure
- **Accuracy: 98.22%** — nearly halved the error rate vs simple network
- **Key lesson**: CNNs outperform flat networks on image data because 
  they preserve spatial relationships between pixels

### Stage 4 — Modern AI
**Sentiment Analysis** (`transform.py`)
- Used Hugging Face `transformers` pipeline (DistilBERT)
- Correctly handled tricky negation: "not bad at all" → POSITIVE (99.98%)

**RAG Pipeline** (`documents.py`)
- Built retrieval-augmented generation from scratch
- Used `sentence-transformers` for semantic embeddings
- Similarity search finds relevant documents by meaning, not keywords
- Question-answering model extracts precise answers from retrieved context

**AI Agent** (`aiagent.py`)
- Built a tool-selecting agent using the ReAct pattern
- Agent reasons about which tool (calculator vs word counter) to use
- Demonstrates: Thought → Action → Observation loop

### Stage 5 — ML Engineering & Deployment
**Production API** (`app.py`)
- Wrapped trained model in a Flask REST API
- Containerized with Docker for reproducible deployment
- Deployed to Render cloud — publicly accessible
- **Key lesson**: train once, serve many times — model loaded once 
  at startup, not retrained per request

## 🛠️ Tech Stack
- **Languages**: Python 3.10
- **ML/Data**: scikit-learn, pandas, numpy, matplotlib
- **Deep Learning**: PyTorch, torchvision
- **NLP/AI**: Hugging Face transformers, sentence-transformers
- **Deployment**: Flask, Docker, Render
- **Version Control**: Git, GitHub

## 📊 Results Summary
| Project | Model | Result |
|---|---|---|
| Titanic survival | Logistic Regression | 80.4% accuracy |
| House prices | Linear Regression | $53,320 avg error |
| Digit recognition (simple) | Neural Network | 96.82% accuracy |
| Digit recognition (CNN) | CNN | 98.22% accuracy |
| Sentiment analysis | DistilBERT | 99.98% confidence |

## 🔧 Running the API locally
```bash
# Clone the repo
git clone https://github.com/Alabamacashing/ml-journey.git
cd ml-journey

# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py

# Test it
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 1, "Age": 25, "Fare": 100, "Sex": 1}'
```

## 🐳 Running with Docker
```bash
docker build -t titanic-api .
docker run -p 5000:5000 titanic-api
```

## 📚 What I Learned
- Data cleaning and exploration are the foundation of every ML project
- Model complexity should match data complexity — neural networks 
  don't always beat simpler models on small, structured datasets
- Feature scaling is critical for distance-based algorithms (k-means, KNN)
- Train/test splits are essential to avoid measuring memorization 
  instead of genuine learning
- Deployment is a completely separate skillset from model building — 
  and equally important for real-world impact
