## 📧 Spam Email Classifier 
A Machine Learning powered Spam Email Classifier built using **TF-IDF, Multinomial Naive Bayes**, deployed as a full-stack web application using **FastAPI, Jinja and Bootstrap 5**.

---

## 🚀 Features

- ✅ Spam vs Real email classification
- ✅ TF-IDF text vectorization
- ✅ Multinomial Naive Bayes model
- ✅ Confidence score with progress bar
- ✅ Clean Bootstrap 5 UI
- ✅ Custom CSS styling
- ✅ FastAPI backend
- ✅ Saved model using Joblib
- ✅ Responsive design

---

## 🧠 Machine Learning Model

- **Algorithm:** Multinomial Naive Bayes
- **Vectorizer:** TF-IDF (max_features=5000)
- **Dataset:** SMS Spam Collection Dataset (Kaggle)
- **Accuracy:** ~96-98%

---

## 🏗️ Project Structure

spam_email_classifier/
│
├── app/
│ ├── main.py 
│ ├── model.py 
│ ├── utils.py
│ ├── templates/
│ │ └── index.html
│ └── static/
│ └── style.css
│
├── data/
│ └── spam.csv
│
├── saved_model/
│ ├── model.pkl
│ └── vectorizer.pkl
│
├── requirements.txt
└── README.md

---

## 📊 Dataset

Dataset used: **SMS Spam Collection Dataset**

Download from Kaggle:
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Extract the zip file and place the spam.csv file inside:
data/spam.csv

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Aayushi-777/spam-email-classifier.git 
cd spam-email-classifier
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

### ▶️ Run the Application
First run:
```bash
python app/model.py
```
which will:
- Train the model
- Save the model in `model.pkl`
- Save the vectorizer in `vectorizer.pkl`

Then run:
```bash
uvicorn app.main:app --reload --port 8002
```
which will start the application

And open the link http://127.0.0.1:8002

---

### 🧩 Technologies used

python
FastAPI
Jinja2
Bootstrap 5
Scikit-learn
Pandas
Joblib
