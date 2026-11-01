# 🛡️ TextShield — AI Fake News & Spam Detector

> NLP-powered text classifier built with Python, Scikit-learn and Streamlit

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8-orange?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat)

## 🎯 What it does
- 🗞️ **Fake News Detector** — Classifies news articles as real or fake
- 📩 **Spam Detector** — Classifies SMS/email messages as spam or legitimate

- 🤖 **AI Explanation** — Claude (Anthropic API) explains in plain English why the text looks fake/real or spam/legit

## 📊 Model Performance
| Task | Algorithm | Accuracy | F1 Score |
|------|-----------|----------|----------|
| Fake News | Logistic Regression | ~99% | 0.991 |
| Spam Detection | Logistic Regression | ~98% | 0.982 |

## 🗃️ Datasets
- **ISOT Fake News Dataset** — 44,898 news articles
- **SMS Spam Collection** — 5,574 messages

## 🛠️ Tech Stack
- **Language:** Python
- **ML:** Scikit-learn, NLTK
- **Vectorization:** TF-IDF (5,000 features, bigrams)
- **Models:** Logistic Regression, Naive Bayes, Linear SVM
- **LLM:** Anthropic API (Claude Haiku) for prediction explanations
- **App:** Streamlit
- **Version Control:** Git & GitHub

## 🚀 How to run locally
```bash
git clone https://github.com/Utkarsh212004/TextShield.git
cd TextShield
pip install -r requirements.txt
streamlit run app.py
```

### Enable the AI explanation (optional)
Create `.streamlit/secrets.toml` (it is git-ignored) and add your key:
```toml
ANTHROPIC_API_KEY = "your-key-here"
```
Without a key the app still works, it just skips the explanation.
On Streamlit Cloud, add the same line under Settings → Secrets.

## 👨‍💻 Built by
**Utkarsh Jain** — MCA AI & Data Science  
JECRC University, Jaipur
