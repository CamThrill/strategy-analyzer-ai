from app import create_app
import os
from flask import request, jsonify
from llm import run_gemini

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


# ── Local dev entrypoint ──────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
