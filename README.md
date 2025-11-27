# Employment Scam Detection 🕵️‍♂️🚫

An NLP-based Machine Learning project to detect fraudulent job advertisements. This project analyzes job descriptions to classify them as "Real" or "Fraudulent," helping to protect job seekers from employment scams.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

## 📌 Project Overview
Employment scams are on the rise, often tricking applicants into providing personal information or money. This project utilizes **Natural Language Processing (NLP)** and **Machine Learning** to identify fake job postings based on textual features.

Key challenges addressed:
- **Imbalanced Dataset:** Fraudulent cases were only ~5% of the data.
- **Text Processing:** Converting unstructured job descriptions into meaningful numerical vectors.

## 📂 Project Structure

├── DataSet.csv              # The dataset containing job descriptions (EMSCAD)
├── Project_Analysis.ipynb   # Main Jupyter Notebook with code, visualizations, and models
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation
‍‍

## 🚀 Key Features & Methodology

### 1. Advanced Preprocessing (NLP)
Instead of simple cleaning, I used **Feature Engineering** within the text:
- **Token Replacement:** Replaced sensitive patterns (Emails, URLs, Phone Numbers, Money amounts) with special tokens like `_EMAIL_`, `_MONEY_` to help the model learn fraud patterns.
- **HTML Removal:** Cleaned raw web-scraped data using `BeautifulSoup`.
- **Lemmatization:** Used NLTK to reduce words to their base root (e.g., "hiring" -> "hire").

### 2. Handling Class Imbalance
The dataset was highly imbalanced (Fraud: ~5%, Real: ~95%).
- **Technique:** Used **SMOTE** (Synthetic Minority Over-sampling Technique) on the training data to generate synthetic samples for the minority class, preventing model bias.

### 3. Vectorization
- **TF-IDF (Term Frequency-Inverse Document Frequency):** Used with **N-grams (1,2)** to capture context (e.g., "wire transfer" vs. just "wire").

### 4. Models Implemented
- **Support Vector Machine (SVM):** Used with a Linear Kernel (best for high-dimensional text data).
- **K-Nearest Neighbors (KNN):** Used with Distance Weighting.

## 📊 Results & Evaluation
Since detecting fraud is the priority, **Recall (Sensitivity)** is the most critical metric (we want to catch all frauds).

| Model | Accuracy | Recall (Sensitivity) | F1-Score | ROC-AUC |
|-------|----------|----------------------|----------|---------|
| **SVM** | **~98%** | **High** | **High** | **~0.99** |
| KNN | Good | Moderate | Moderate | ~0.90 |

> **Conclusion:** SVM outperformed KNN significantly in detecting fraudulent cases, making it the preferred model for this task.

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Employment-Scam-Detection.git
   cd Employment-Scam-Detection

### Create a Virtual Environment (Optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### Install Dependencies
```bash
pip install -r requirements.txt

### Run the Analysis
Open `Project_Analysis.ipynb` in Jupyter Notebook or VS Code to see the step-by-step implementation.

## 📦 Libraries Used
- **Pandas & NumPy:** Data manipulation.
- **Matplotlib & Seaborn:** Visualization (Distribution plots, Confusion Matrix).
- **NLTK:** Text processing (Stopwords, Lemmatizer).
- **Scikit-Learn:** ML models and evaluation metrics.
- **Imbalanced-Learn:** SMOTE implementation.

## 🤝 Future Improvements
- Implement **Deep Learning** models like LSTM or BERT for better context understanding.
- Incorporate non-text features (e.g., "Has Company Logo?", "Has Questions?").
- Deploy the model as a web API using Flask or FastAPI.