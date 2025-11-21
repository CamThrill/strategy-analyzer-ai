from app import create_app
import os
from flask import request, jsonify
from llm import run_gemini
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Create the Flask app from the factory
app = create_app()


# ── Gemini test endpoint ─────────────────────────
@app.route("/api/test_llm", methods=["POST"])
def api_test_llm():
    """
    Simple endpoint to verify Gemini integration.
    Send JSON: { "prompt": "your question" }
    """
    data = request.get_json() or {}
    user_prompt = data.get(
        "prompt",
        "Give a short one-sentence confirmation that the Gemini API is connected.",
    )

    try:
        response_text = run_gemini(user_prompt)
        return jsonify({"ok": True, "response": response_text})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ── LLM: Data Preparation description ───────────────────────────────────────────
@app.route("/api/llm/data_preparation", methods=["POST"])
def llm_data_preparation():
    """
    Generate a written explanation of the data-preparation steps for the report.
    Expects JSON like:
    {
      "columns": ["profit", "entry_time", "exit_time", "market_position", ...],
      "steps": ["normalized profit", "encoded market_position", "derived trade_duration"],
      "extra_context": "Any notes you want"
    }
    """
    data = request.get_json() or {}
    columns = data.get("columns", [])
    steps = data.get("steps", [])
    extra = data.get("extra_context", "")

    prompt = f"""
You are helping write the Methodology / Pipeline section for a machine learning project
on trading strategy performance.

Write 1–2 detailed paragraphs describing the DATA PREPARATION steps that were performed.

Columns in the raw dataset:
{columns}

Preprocessing steps that were actually implemented:
{steps}

Extra context:
{extra}

Use clear academic/project-report style language, and mention:
- cleaning, encoding, splitting
- why these steps matter for model performance
- that the steps were informed by an LLM review of the raw columns.
"""
    text = run_gemini(prompt)
    return jsonify({"ok": True, "text": text})
    

# ── LLM: Model Training & Testing description ───────────────────────────────────
@app.route("/api/llm/model_training", methods=["POST"])
def llm_model_training():
    """
    Generate a description of model training/testing for the report.
    Expects JSON like:
    {
      "target": "profitable_trade",
      "features": ["profit_norm", "duration", "market_position_long", ...],
      "model_type": "Logistic Regression",
      "train_test_split": "80/20 with random_state=42",
      "extra_context": "Any notes"
    }
    """
    data = request.get_json() or {}
    target = data.get("target", "")
    features = data.get("features", [])
    model_type = data.get("model_type", "Logistic Regression")
    split = data.get("train_test_split", "80/20 train-test split")
    extra = data.get("extra_context", "")

    prompt = f"""
You are writing the MODEL TRAINING AND TESTING subsection for a project report.

Describe in 1–2 paragraphs how the model was trained and evaluated.

Details:
- Prediction target: {target}
- Input features: {features}
- Model type: {model_type}
- Train/test strategy: {split}
- Extra context: {extra}

Explain:
- why this model is suitable for the task
- the rationale for the chosen train/test split
- that an LLM was consulted to suggest appropriate models and validation strategy.
"""
    text = run_gemini(prompt)
    return jsonify({"ok": True, "text": text})


# ── LLM: Evaluation Metrics explanation ─────────────────────────────────────────
@app.route("/api/llm/eval_metrics", methods=["POST"])
def llm_eval_metrics():
    """
    Interpret evaluation metrics (accuracy/precision/recall/F1, confusion matrix).
    Expects JSON like:
    {
      "metrics": {
        "accuracy": 0.87,
        "precision": 0.82,
        "recall": 0.79,
        "f1": 0.80
      },
      "confusion_matrix": [[50, 10],[8, 32]],
      "labels": ["losing_or_break_even", "profitable"],
      "extra_context": "Any notes about behavior"
    }
    """
    data = request.get_json() or {}
    metrics = data.get("metrics", {})
    cm = data.get("confusion_matrix", [])
    labels = data.get("labels", [])
    extra = data.get("extra_context", "")

    prompt = f"""
You are helping with the EVALUATION METRICS section of a machine learning project report.

Metrics:
{metrics}

Confusion matrix (rows = true labels, columns = predicted labels):
{cm}

Label order:
{labels}

Extra context:
{extra}

Write 1–2 paragraphs that:
- explain what the accuracy, precision, recall, and F1-score indicate
- describe which classes the model predicts well and which are confused
- identify any trade-offs (e.g., high precision but lower recall)
- mention that an LLM was used to help interpret the confusion matrix and scores.
"""
    text = run_gemini(prompt)
    return jsonify({"ok": True, "text": text})


# ── LLM: Feature selection / hyperparameter guidance (optional) ─────────────────
@app.route("/api/llm/feature_selection", methods=["POST"])
def llm_feature_selection():
    """
    Get LLM suggestions for feature selection or hyperparameter tuning.
    Expects JSON like:
    {
      "all_features": ["profit_norm", "duration", "volatility", ...],
      "target": "profitable_trade",
      "model_type": "Logistic Regression",
      "extra_context": "Any info about overfitting/underfitting"
    }
    """
    data = request.get_json() or {}
    all_features = data.get("all_features", [])
    target = data.get("target", "")
    model_type = data.get("model_type", "Logistic Regression")
    extra = data.get("extra_context", "")

    prompt = f"""
You are advising on FEATURE SELECTION and basic HYPERPARAMETER TUNING
for a trading ML project.

Prediction target:
{target}

Candidate features:
{all_features}

Model type:
{model_type}

Extra context:
{extra}

Give:
1) A ranked list of the most important features to try first, with short justifications.
2) Simple, concrete hyperparameter ranges or settings to explore (not a full grid search).
3) 1 short paragraph suitable for a report explaining that an LLM was consulted to
   propose feature importance and tuning directions.
"""
    text = run_gemini(prompt)
    return jsonify({"ok": True, "text": text})



@app.route("/api/llm/strategy_combo", methods=["POST"])
def llm_strategy_combo():
    """
    Ask the LLM to suggest a combination of strategies with low drawdown.

    Expects JSON like:
    {
      "strategies": [
        {
          "name": "Strat A",
          "net_profit": 12000,
          "max_drawdown": -2500,
          "sharpe": 1.4,
          "win_rate": 62.0
        },
        ...
      ],
      "max_strategies": 3
    }
    """
    data = request.get_json() or {}
    strategies = data.get("strategies", [])
    max_strats = data.get("max_strategies", 3)

    if not strategies:
        return jsonify({"ok": False, "error": "No strategies provided"}), 400

    prompt = f"""
You are helping select a portfolio of trading strategies.

Each strategy has metrics including net profit, max drawdown, Sharpe, and win rate.

Your task:
- Choose up to {max_strats} strategies that form a good portfolio
- Primary objective: minimize overall drawdown
- Secondary objectives: keep reasonable total net profit and decent Sharpe
- Avoid picking multiple strategies that look very weak or redundant

Here is the list of strategies and their metrics (JSON):

{strategies}

Respond with PURE JSON, NO MARKDOWN, NO BACKTICKS, NO EXPLANATION TEXT.
The JSON must have exactly this shape:

{{
  "selected_strategies": ["name1", "name2"],
  "rationale": "1–2 short paragraphs explaining why these were chosen."
}}
"""

    text = run_gemini(prompt)

    # --- Clean up common markdown / fencing so we can parse JSON safely ---
    cleaned = text.strip()

    # Strip ```json ... ``` or ``` ... ``` fences if present
    if cleaned.startswith("```"):
        # Remove leading ```json or ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        # Truncate at closing ```
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        return jsonify({"ok": True, "selection": parsed})
    except Exception:
        # Fallback: send raw text so UI can at least display something
        return jsonify({"ok": True, "raw_text": text})


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

@app.route("/api/ml/evaluate", methods=["POST"])
def api_ml_evaluate():
    """
    Computes core evaluation metrics and confusion matrix for the trading classifier.
    Expects:
    {
        "y_true": [...],
        "y_pred": [...]
    }
    """
    data = request.get_json() or {}
    y_true = data.get("y_true", [])
    y_pred = data.get("y_pred", [])

    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return jsonify({"ok": False, "error": "Invalid y_true/y_pred"}), 400

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return jsonify({
        "ok": True,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        },
        "confusion_matrix": cm.tolist()
    })



# ── Local dev entrypoint ──────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
