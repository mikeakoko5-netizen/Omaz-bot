from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import urllib.request
import urllib.error

app = Flask(__name__)
CORS(app)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "PO AI Bot Server running! Powered by Gemini AI"})
@app.route("/geminitest")
def geminitest():
    try:
        payload = json.dumps({
            "contents": [{
                "parts": [{"text": "Say OK"}]
            }]
        }).encode()

        req = urllib.request.Request(
            GEMINI_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode()

    except urllib.error.HTTPError as e:
        return e.read().decode(), e.code

@app.route("/analyze", methods=["POST"])
def analyze():
    global GEMINI_URL
    GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={os.environ.get('GEMINI_API_KEY')}"
    try:
        data = request.json
        market_data = data.get("market_data", {})
        pair = data.get("pair", "EUR/USD")
        pair_type = data.get("pair_type", "forex")

        prompt = f"""You are an expert trader analyzing {pair} ({'OTC' if pair_type == 'otc' else 'Forex'}) for a short-term binary option trade on Pocket Option platform.

Current Market Data for {pair}:
- Price: {market_data.get('price', 'N/A')}
- Trend: {market_data.get('trend', 'N/A')}
- EMA9: {market_data.get('ema9', 'N/A')}
- EMA21: {market_data.get('ema21', 'N/A')}
- EMA50: {market_data.get('ema50', 'N/A')}
- RSI(14): {market_data.get('rsi', 'N/A')}
- Stochastic: {market_data.get('stoch', 'N/A')}
- MACD Histogram: {market_data.get('macd_hist', 'N/A')}
- Bollinger Upper: {market_data.get('bb_upper', 'N/A')}
- Bollinger Lower: {market_data.get('bb_lower', 'N/A')}
- Bollinger Mid: {market_data.get('bb_mid', 'N/A')}
- ATR: {market_data.get('atr', 'N/A')}
- 20-candle High: {market_data.get('high20', 'N/A')}
- 20-candle Low: {market_data.get('low20', 'N/A')}
- Session: {market_data.get('session', 'N/A')}

Last 10 candles (oldest to newest):
{market_data.get('candle_summary', 'N/A')}

{'Note: OTC pair - broker-generated price. Focus on momentum and oscillator signals.' if pair_type == 'otc' else 'Note: Real forex pair - trend and momentum signals are reliable.'}

Analyze this data and provide a trading signal for a short-term binary option.

Respond ONLY in this exact JSON format, no markdown, no extra text:
{{
  "direction": "CALL",
  "confidence": 70,
  "strength": "MODERATE",
  "signal_type": "TREND",
  "bull_score": 6,
  "bear_score": 3,
  "summary": "2-3 sentence explanation here",
  "key_reasons": ["reason 1", "reason 2", "reason 3"],
  "risk_note": "one sentence about risk",
  "recommended_expiry": "3"
}}

Rules:
- direction must be exactly "CALL" or "PUT"
- confidence must be a number between 50 and 85
- strength must be "STRONG", "MODERATE" or "WEAK"
- signal_type must be "TREND", "REVERSAL" or "MOMENTUM"
- recommended_expiry must be "1", "3" or "5"
- If market is unclear or choppy, set confidence below 55
- Be honest and realistic"""

        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200}
        }).encode()

        req = urllib.request.Request(
            GEMINI_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        text = result["candidates"][0]["content"]["parts"][0]["text"]
        clean = text.replace("```json", "").replace("```", "").strip()
        signal = json.loads(clean)
        return jsonify({"success": True, "signal": signal})

    except json.JSONDecodeError as e:
        return jsonify({"success": False, "error": f"JSON parse error: {str(e)}"}), 500
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return jsonify({"success": False, "error": f"Gemini API error {e.code}: {body}"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/telegram", methods=["POST"])
def telegram():
    try:
        data = request.json
        token = data.get("token")
        chat_id = data.get("chat_id")
        text = data.get("text")
        if not all([token, chat_id, text]):
            return jsonify({"ok": False, "error": "Missing token, chat_id or text"}), 400
        url = f"https://api.telegram.org/bot{token}/sendMessage"
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
