import os
import google.generativeai as genai

# Get API key from env (Cloud Run gets it from Secret Manager)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. "
        "For local dev, run: export GEMINI_API_KEY='your-key-here'. "
        "In Cloud Run, make sure the secret is wired to this env var."
    )

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Use a current, widely available model name for AI Studio keys
DEFAULT_MODEL = "gemini-2.0-flash"


def run_gemini(prompt: str, system_instruction: str | None = None) -> str:
    """
    Simple helper to send a prompt to Gemini and return the text response.
    """
    model = genai.GenerativeModel(
        model_name=DEFAULT_MODEL,
        system_instruction=system_instruction
        or "You are an expert ML and quantitative trading assistant. "
           "Give concise, technically accurate answers suitable for a project report."
    )

    # For the current google-generativeai SDK, a plain string is fine
    response = model.generate_content(prompt)

    # Some safety in case there's no text field
    return (getattr(response, "text", "") or "").strip()
