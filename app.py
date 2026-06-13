from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
import urllib.request
import urllib.error

app = Flask(__name__)
CORS(app)

GEMINI_MODEL = "gemini-2.0-flash"


def get_gemini_url():
    key = os.environ.get("GEMINI_API_KEY", "")
    return "https://generativelanguage.googleapis.com/v1beta/models/" + GEMINI_MODEL + ":generateContent?key=" + key


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "PO AI Bot Server running! Powered by Gemini AI"})


def build_prompt(market_data, pair, pair_type):
    note = "Note: OTC pair - broker-generated price. Focus on momentum and oscillator signals."
    if pair_type != "otc":
        note = "Note: Real forex pair - trend and momentum signals are reliable."

    prompt = "You are an expert trader analyzing " + pair
    prompt += " (" + ("OTC" if pair_type == "otc" else "Forex") + ")"
    prompt += " for a short-term binary option trade on Pocket Option platform.\n\n"
    prompt += "Current Market Data for " + pair + ":\n"
    prompt += "- Price: " + str(market_data.get("price", "N/A")) + "\n"
    prompt += "- Trend: " + str(market_data.get("trend", "N/A")) + "\n"
    prompt += "- EMA9: " + str(market_data.get("ema9", "N/A")) + "\n"
    prompt += "- EMA21: " + str(market_data.get("ema21", "N/A")) + "\n"
    prompt += "- EMA50: " + str(market_data.get("ema50", "N/A")) + "\n"
    prompt += "- RSI(14): " + str(market_data.get("rsi", "N/A")) + "\n"
    prompt += "- Stochastic: " + str(market_data.get("stoch", "N/A")) + "\n"
    prompt += "- MACD Histogram: " + str(market_data.get("macd_hist", "N/A")) + "\n"
    prompt += "- Bollinger Upper: " + str(market_data.get("bb_upper", "N/A")) + "\n"
    prompt += "- Bollinger Lower: " + str(market_data.get("bb_lower", "N/A")) + "\n"
    prompt += "- Bollinger Mid: " + str(market_data.get("bb_mid", "N/A")) + "\n"
    prompt += "- ATR: " + str(market_data.get("atr", "N/A")) + "\n"
    prompt += "- 20-candle High: " + str(market_data.get("high20", "N/A")) + "\n"
    prompt += "- 20-candle Low: " + str(market_data.get("low20", "N/A")) + "\n"
    prompt += "- Session: " + str(market_data.get("session", "N/A")) + "\n\n"
    prompt += "Last 10 candles (oldest to newest):\n"
    prompt += str(market_data.get("candle_summary", "N/A")) + "\n\n"
    prompt += note + "\n\n"
    prompt += "Analyze this data and provide a trading signal for a short-term binary option.\n\n"
    prompt += "Respond with ONLY a single-line JSON object. No markdown. No code fences. No explanations before or after. "
    prompt += "No line breaks inside string values.\n"
    prompt += "Use this exact structure and key names:\n"
    prompt += '{"direction": "CALL", "confidence": 70, "strength": "MODERATE", "signal_type": "TREND", "bull_score": 6, "bear_score": 3, "summary": "short explanation", "key_reasons": ["reason 1", "reason 2", "reason 3"], "risk_note": "short risk note", "recommended_expiry": "3"}\n\n'
    prompt += "Rules:\n"
    prompt += '- direction must be exactly "CALL" or "PUT"\n'
    prompt += "- confidence must be a number between 50 and 85\n"
    prompt += '- strength must be "STRONG", "MODERATE" or "WEAK"\n'
    prompt += '- signal_type must be "TREND", "REVERSAL" or "MOMENTUM"\n'
    prompt += '- recommended_expiry must be "1", "3" or "5"\n'
    prompt += "- summary and risk_note must be plain short sentences with NO line breaks, NO quotes inside them\n"
    prompt += "- If market is unclear or choppy, set confidence below 55\n"
    prompt += "- Be honest and realistic\n"
    prompt += "- Your entire reply must be ONLY the JSON object, starting with { and ending with }"
    return prompt


def extract_json(text):
    if not text or not text.strip():
        raise ValueError("Empty text from AI")
    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        candidate = match.group(0)
        candidate = candidate.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        return json.loads(candidate)
    raise ValueError("No JSON object found in AI response")


@app.route("/analyze", methods=["POST"])
def analyze():
    raw_text = ""
    try:
        data = request.json
        market_data = data.get("market_data", {})
        pair = data.get("pair", "EUR/USD")
        pair_type = data.get("pair_type", "forex")

        prompt = build_prompt(market_data, pair, pair_type)

        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1500
            }
        }).encode()

        req = urllib.request.Request(
            get_gemini_url(),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        candidates = result.get("candidates", [])
        if not candidates:
            feedback = result.get("promptFeedback", {})
            return jsonify({
                "success": False,
                "error": "No candidates returned by Gemini. Feedback: " + json.dumps(feedback)
            }), 500

        candidate0 = candidates[0]
        finish_reason = candidate0.get("finishReason", "")
        parts = candidate0.get("content", {}).get("parts", [])

        if not parts:
            return jsonify({
                "success": False,
                "error": "Empty response from Gemini. finishReason=" + str(finish_reason)
            }), 500

        raw_text = parts[0].get("text", "")

        try:
            signal = extract_json(raw_text)
        except Exception as parse_err:
            return jsonify({
                "success": False,
                "error": "JSON parse error: " + str(parse_err) + " (finishReason=" + str(finish_reason) + ")",
                "raw": raw_text[:500]
            }), 500

        return jsonify({"success": True, "signal": signal})

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return jsonify({"success": False, "error": "Gemini API error " + str(e.code) + ": " + body}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "raw": raw_text[:500]}), 500


@app.route("/telegram", methods=["POST"])
def telegram():
    try:
        data = request.json
        token = data.get("token")
        chat_id = data.get("chat_id")
        text = data.get("text")
        if not all([token, chat_id, text]):
            return jsonify({"ok": False, "error": "Missing token, chat_id or text"}), 400
        url = "https://api.telegram.org/bot" + token + "/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": text, "parse_mode": "HTML"}).encode()
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
