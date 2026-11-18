# 🕵️ Fake Job Posting Detection

A machine-learning-based Streamlit application for detecting potentially fraudulent job postings.

Live Demo: https://fakejobpostingdetectionproject-tobyrjmebw2dvle72vzcxk.streamlit.app/
Github Repository: https://github.com/nancy-2124/Fake_Job_Posting_Detection_Project/edit/main/fake_job_posting_detection/

## Features
- Fake job posting prediction
- Naive Bayes, Logistic Regression and numerical models
- Combined prediction
- Exploratory Data Analysis
- Fraudulent vs genuine distribution
- Location and industry analysis
- Correlation heatmap
- Word clouds
- Confusion matrices
- CSV upload

## Technologies
Python, Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, WordCloud.

## Project Structure

```text
Fake-Job-Posting-Detection/
├── app.py
├── fake_job_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── Fake_job__posting_detection.ipynb
└── model.ipynb
```

## Run Locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deploy the repository using Streamlit Community Cloud and select `app.py` as the main file.

## Disclaimer

Predictions are model-based indications and should not be treated as a guarantee that a job posting is genuine or fraudulent.
