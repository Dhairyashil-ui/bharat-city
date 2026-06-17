# -------------------------------
# 1. Imports
# -------------------------------
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler

# TensorFlow import (graceful fallback if unavailable)
try:
    from tensorflow.keras.models import load_model
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# Optional: MySQL
try:
    import mysql.connector
    _MYSQL_AVAILABLE = True
except ImportError:
    _MYSQL_AVAILABLE = False

# Optional: Groq AI
try:
    from dotenv import load_dotenv
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# -------------------------------
# 2. Logging
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------
# 3. Create Flask App
# -------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------
# 4. Environment / .env
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
if _GROQ_AVAILABLE:
    load_dotenv(BASE_DIR / ".env")
    _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY") or "")
else:
    _groq_client = None

# -------------------------------
# 5. Database Connection (optional — app runs without MySQL)
# -------------------------------
db = None
cursor = None
if _MYSQL_AVAILABLE:
    try:
        db = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "smart_energy"),
        )
        cursor = db.cursor()
        logger.info("MySQL connected successfully")
    except Exception as e:
        logger.warning("MySQL unavailable (non-fatal): %s", e)
        db = None
        cursor = None

# -------------------------------
# 6. Load Model + Scaler
# -------------------------------
SEQ_LENGTH = 24

# CSV may live in backend/ or in the project root (one level up)
_csv_backend = BASE_DIR / "AEP_hourly.csv"
_csv_root    = BASE_DIR.parent / "AEP_hourly.csv"
CSV_PATH     = _csv_backend if _csv_backend.exists() else _csv_root

MODEL_PATH = BASE_DIR / "lstm_model.h5"

scaler = MinMaxScaler()
model  = None

try:
    df = pd.read_csv(CSV_PATH)
    _data = df["AEP_MW"].values.reshape(-1, 1)
    scaler.fit(_data)
    logger.info("Scaler fitted from %s (%d rows)", CSV_PATH, len(df))
except Exception as e:
    logger.error("Failed to load CSV / fit scaler: %s", e)

if _TF_AVAILABLE and MODEL_PATH.exists():
    try:
        model = load_model(str(MODEL_PATH), compile=False)
        logger.info("LSTM model loaded from %s", MODEL_PATH)
    except Exception as e:
        logger.error("Failed to load LSTM model: %s", e)
        model = None
else:
    if not _TF_AVAILABLE:
        logger.warning("TensorFlow not installed — predictions will use fallback mean")
    elif not MODEL_PATH.exists():
        logger.warning("Model file not found at %s — predictions will use fallback mean", MODEL_PATH)


def get_ai_suggestion(prediction: float, appliance_data) -> str:
    """Query Groq LLM for energy optimization advice. Returns plain text."""
    if not _GROQ_AVAILABLE or _groq_client is None:
        return "AI suggestion unavailable: groq package not installed."
    if not os.getenv("GROQ_API_KEY"):
        return "AI suggestion unavailable: add GROQ_API_KEY to .env"
    try:
        if isinstance(appliance_data, dict) and appliance_data:
            lines = "\n".join(f"- {k}: {v} kWh" for k, v in appliance_data.items())
        else:
            lines = str(appliance_data) if appliance_data else "(no appliance breakdown provided)"

        prompt = f"""You are a smart home energy optimization AI.

Energy prediction: {prediction:.2f} MW

Appliance usage:
{lines}

Analyze and return:
1. Total consumption insight
2. Which appliance consumes most energy
3. Wastage detection
4. Practical suggestions to reduce energy
5. Cost saving tips

Be specific. Mention appliances like AC, fridge, etc.
Keep answer short and actionable."""

        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("Groq AI call failed: %s", e)
        return f"AI Error: {e}"


# -------------------------------
# 7. Routes
# -------------------------------

@app.route("/")
def home():
    return jsonify({"status": "ok", "service": "Smart Energy API", "model_loaded": model is not None})


def _predict_mw(raw_input) -> float:
    """Run LSTM inference. raw_input must be a flat array of exactly SEQ_LENGTH floats."""
    arr = np.array(raw_input, dtype=float).reshape(-1, 1)
    input_scaled = scaler.transform(arr).reshape(1, SEQ_LENGTH, 1)
    if model is not None:
        pred_scaled = model.predict(input_scaled, verbose=0)
        return float(scaler.inverse_transform(pred_scaled)[0][0])
    # Fallback: mean of input (when model / TF unavailable)
    return float(np.mean(arr))


def _validate_input(body: dict):
    """Return (input_array_1D, error_response_tuple_or_None)."""
    input_data = body.get("input")
    if input_data is None:
        return None, (jsonify({"error": "Missing 'input' field in request body"}), 400)

    try:
        input_data = [float(v) for v in input_data]
    except (TypeError, ValueError):
        return None, (jsonify({"error": "'input' must be a list of numbers"}), 400)

    if len(input_data) < SEQ_LENGTH:
        return None, (
            jsonify({"error": f"At least {SEQ_LENGTH} input values required, got {len(input_data)}"}),
            400,
        )

    # Take exactly SEQ_LENGTH values (pad or trim)
    input_data = input_data[:SEQ_LENGTH]
    return np.array(input_data, dtype=float), None


# ---------- Prediction route ----------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json(silent=True) or {}
        logger.info("POST /predict — payload keys: %s", list(body.keys()))

        arr, err = _validate_input(body)
        if err:
            return err

        pred_actual = _predict_mw(arr)
        logger.info("Prediction result: %.2f MW", pred_actual)

        return jsonify({"prediction_MW": pred_actual})

    except Exception as e:
        logger.exception("Error in /predict")
        return jsonify({"error": str(e)}), 500


# ---------- Optimization route ----------
@app.route("/optimize", methods=["POST"])
def optimize():
    try:
        body = request.get_json(silent=True) or {}
        logger.info("POST /optimize — payload keys: %s", list(body.keys()))

        arr, err = _validate_input(body)
        if err:
            return err

        appliance_data = body.get("appliances") or {}
        pred_actual = _predict_mw(arr)

        if pred_actual < 14000:
            status = "Low usage"
            suggestion = "Energy usage is optimal. No action needed."
        elif pred_actual < 15500:
            status = "Moderate usage"
            suggestion = "Try reducing non-essential appliances during peak hours."
        else:
            status = "High usage"
            suggestion = "Reduce AC/heater usage and shift heavy loads to off-peak hours."

        ai_suggestion = get_ai_suggestion(pred_actual, appliance_data)

        # Persist to database (best-effort)
        if cursor is not None and db is not None:
            try:
                cursor.execute(
                    """
                    INSERT INTO predictions (input_data, prediction, status, suggestion)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (str(arr.tolist()), pred_actual, status, suggestion),
                )
                db.commit()
            except Exception as e:
                logger.warning("DB write failed (non-fatal): %s", e)

        logger.info("Optimize result: %.2f MW — %s", pred_actual, status)

        return jsonify({
            "prediction_MW": pred_actual,
            "status": status,
            "suggestion": suggestion,
            "ai_suggestion": ai_suggestion,
        })

    except Exception as e:
        logger.exception("Error in /optimize")
        return jsonify({"error": str(e)}), 500


# ---------- History route ----------
@app.route("/history", methods=["GET"])
def history():
    if cursor is None:
        return jsonify([])
    try:
        cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10")
        rows = cursor.fetchall()
        return jsonify([
            {
                "id": row[0],
                "input_data": row[1],
                "prediction": float(row[2]) if row[2] is not None else None,
                "status": row[3],
                "suggestion": row[4],
                "time": str(row[5]),
            }
            for row in rows
        ])
    except Exception as e:
        logger.error("History query failed: %s", e)
        return jsonify([])


# -------------------------------
# 8. Run App
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Flask on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=False)