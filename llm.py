import os
import google.generativeai as genai


# Read the key from environment (from Secret Manager in Cloud Run)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    # local / debugging
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "Make sure the secret is configured in Cloud Run and in your local .env if testing locally."
    )


# Configure the Gemini client
genai.configure(api_key=GEMINI_API_KEY)


# You can change model later
DEFAULT_MODEL = "gemini-1.5-flash"


def run_gemini(prompt: str, system_instruction: str | None = None) -> str:
    """
    Simple helper to send a prompt to Gemini and return the text response.
    We'll reuse this for:
      - Data prep suggestions
      - Model training/testing design
      - Metric interpretation
      - Feature selection / hyperparameter hints
    """
    model = genai.GenerativeModel(
        model_name=DEFAULT_MODEL,
        system_instruction=system_instruction or (
            "You are an expert quantitative trading and machine learning assistant. "
            "Give concise, technically accurate answers suitable for a project report."
        ),
    )

    response = model.generate_content(prompt)
    # Gemini SDK returns a response object; .text gives concatenated text
    return response.text.strip()
