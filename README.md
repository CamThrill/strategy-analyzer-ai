# strategy-analyzer-cloudrun

# 📊 Strategy Analyzer + AI Assistant  
CAP 4630 – Intro to Artificial Intelligence  
**Final Project – Fall 2025**

---

## 🔍 Overview  
The **Strategy Analyzer** is an interactive AI-powered tool for analyzing algorithmic trading strategies.  
Users can upload multiple **NinjaTrader `.xlsx` backtest files**, visualize performance, compare strategies, and use AI assistance (Google Gemini Flash 2.0) to generate:

- Data preparation explanations  
- Model training & testing descriptions  
- Evaluation metrics interpretations  
- Feature selection & hyperparameter tuning suggestions  
- AI-selected strategy combinations optimized for lowest drawdown  

This project integrates **Python, Flask, Plotly, and Google Gemini AI**, and is designed to run easily in **GitHub Codespaces**.

---

## 🚀 Features

### ✔️ Upload & Analyze Multiple Backtests  
Automatically processes each uploaded `.xlsx` file and computes:

- Win Rate  
- Sharpe Ratio  
- Sortino Ratio  
- Max Drawdown  
- Total Net Profit  
- Required Capital  
- Per-Strategy Metrics  
- Portfolio Equity Curves  
- Correlation Matrix  

---

### 🎨 Interactive Visualizations (Plotly)

- Cumulative PnL Chart  
- Correlation Heatmap  
- Per-Strategy Metrics Table  
- ML Evaluation Results (Accuracy, Precision, Recall, F1)  
- Confusion Matrix Heatmap  

---

### 🤖 AI Pipeline Assistant (Gemini Flash 2.0)

Generates ready-to-use text for sections such as:

- Data Preparation  
- Model Training & Testing  
- Evaluation Metrics  
- Feature Selection & Hyperparameter Tuning  
- Strategy Combination Optimization (lowest drawdown portfolio)  

Perfect for AI-course reports and presentations.

---

## 🔧 Installation & Setup (GitHub Codespaces Recommended)

Open the repository in **GitHub Codespaces**, then run:

### 1️⃣ Create & activate virtual environment

python3 -m venv venv \\
source venv/bin/activate

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Set your API key

Each user generates their own Gemini API key.
My key is NOT included.

Generate your key here:
https://aistudio.google.com/app/apikey

Then export it manually:

export GOOGLE_API_KEY="your_gemini_api_key_here" (No "" just the key) \\
export ENV=dev

### 4️⃣ Run the application

python run.py
