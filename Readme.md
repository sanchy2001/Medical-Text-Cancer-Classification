# 🏥 Medical Text Dataset — Cancer Document Classification

## 📌 Project Overview
An NLP and Deep Learning project to automatically classify medical text documents into cancer types — **Thyroid Cancer**, **Colon Cancer**, and **Lung Cancer**. By leveraging multiple text classification models, this project helps healthcare professionals and researchers quickly identify cancer-related document categories from unstructured medical text.

---

## 🎯 Problem Statement
- Classify medical text into 3 cancer categories (Thyroid, Colon, Lung)
- Handle unstructured free-form medical language
- Compare traditional ML vs. deep learning NLP approaches
- Build a scalable system that adapts to evolving medical knowledge

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Total Records** | 7,570 rows |
| **Features** | 3 columns (ID, Cancer Type, Medical Text) |
| **Target Classes** | Thyroid Cancer (2,810) · Colon Cancer (2,579) · Lung Cancer (2,180) |

> ⚠️ **Note:** The dataset (`Medical.csv`) is not included in this repository due to GitHub's 100MB file size limit (file size: ~180MB).
> 
> 📥 **Download the dataset here:** [Medical.csv — Google Drive](https://drive.google.com/file/d/1eyKtPRjGCJX6YXSlThBJG116R3EFvD26/view?usp=sharing)
> 
> After downloading, place it in the `data/` folder before running the notebook.

---

## 🔧 Tech Stack

- **Language:** Python 3
- **NLP Libraries:** NLTK, Keras Tokenizer, Keras Preprocessing
- **ML Libraries:** Scikit-learn, TensorFlow, Keras
- **Deep Learning:** LSTM, Bidirectional LSTM, SimpleRNN, Embedding layers
- **Utilities:** Pandas, NumPy, Matplotlib
- **Environment:** Jupyter Notebook

---

## 📁 Project Structure

```
MedicalTextClassification/
│
├── data/
│   └── Medical.csv              ← raw medical text dataset
│
├── notebooks/
│   └── Medical_text_data.ipynb  ← main project notebook
│
├── README.md
└── .gitignore
```

---

## 🔄 Project Workflow

### Phase 1 — Data Understanding
- Loaded 7,570 medical text records across 3 cancer categories
- Explored class distribution (balanced across all 3 classes)
- Identified and handled null values

### Phase 2 — Text Preprocessing
- Converted all text to lowercase
- Removed punctuation and English stop words (NLTK)
- Built a custom `text_process()` function for cleaning
- Created Term Document Matrix (TDM) using `CountVectorizer`
- Vocabulary size after preprocessing: **208,269 unique tokens**

### Phase 3 — Deep Learning Preprocessing
- Tokenized text using Keras `Tokenizer` (top 8,000 words selected)
- Padded sequences to a fixed length of 100 tokens
- Applied word `Embedding` layer (embedding size = 100)
- One-hot encoded target labels using `to_categorical`

### Phase 4 — Modelling
- Trained 4 models ranging from traditional ML to advanced deep learning
- Evaluated using Accuracy, Confusion Matrix, and Classification Report
- 80/20 train-test split applied across all models

---

## 📈 Model Results

| Model | Accuracy |
|---|---|
| Naive Bayes | 92.73% |
| Simple RNN | 96.43% |
| LSTM | 98.41% |
| **Bidirectional LSTM** | **98.48% ✅** |

> **Best Model:** Bidirectional LSTM with 98.48% accuracy

---

## 🏗️ Model Architectures

### Naive Bayes
- `MultinomialNB` on TDM (Bag-of-Words) features
- Fast baseline with strong performance at 92.73%

### Simple RNN
- `Embedding → SimpleRNN → Dense(3, softmax)`
- Adam optimizer, categorical cross-entropy loss

### LSTM
- `Embedding → LSTM(128) → Dropout(0.2) → Dense(3, softmax)`
- Captures long-range text dependencies
- Trained for 8 epochs, batch size 32

### Bidirectional LSTM ✅
- `Embedding → Bidirectional(LSTM(65)) → Dropout(0.2) → Dense(3, softmax)`
- Processes text in both forward and backward directions
- Captures richer contextual information for classification
- Trained for 5 epochs, batch size 32

---

## 🔍 Key NLP Concepts Applied

- **Stop Word Removal** — Reduces noise, improves model focus on meaningful terms
- **Punctuation Removal** — Normalizes raw text for tokenization
- **Bag-of-Words (TDM)** — Sparse matrix representation for Naive Bayes
- **Word Embeddings** — Dense vector representation for deep learning models
- **Sequence Padding** — Ensures uniform input length for RNN-based models

---

## 💡 Key Findings & Recommendations

- Deep learning models significantly outperform traditional ML for medical text
- Bidirectional context (BiLSTM) is critical for understanding complex medical language
- Larger vocabulary sizes improve accuracy but increase training time
- Dropout regularization prevents overfitting on medical terminology
- Pre-trained medical word embeddings (e.g., BioBERT) could push accuracy even higher

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn tensorflow keras nltk matplotlib
   ```

3. **Download NLTK stop words**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. **Download and place the dataset**
   - Download `Medical.csv` from [Google Drive](https://drive.google.com/file/d/1eyKtPRjGCJX6YXSlThBJG116R3EFvD26/view?usp=sharing)
   - Place it in the `data/` folder

5. **Open and run the notebook**
   ```bash
   jupyter notebook notebooks/Medical_text_data.ipynb
   ```

---

## 🎯 Conclusion

This project demonstrates the power of NLP and deep learning for automated classification of cancer-related medical documents. The **Bidirectional LSTM** model achieved **98.48% accuracy**, making it highly reliable for distinguishing between Thyroid, Colon, and Lung cancer text. Such a system can provide valuable support to healthcare professionals, researchers, and patients in navigating and understanding unstructured medical literature.

---

## 👤 Author

**Vijay** *(as referenced in project notebook)*

---

> *"Analyzing and classifying these types of text data can provide valuable insights for healthcare professionals, researchers, and patients, facilitating better understanding, diagnosis, treatment, and management of cancer."*