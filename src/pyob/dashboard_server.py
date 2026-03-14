import logging
import os
import signal

from flask import Flask, jsonify, render_template

app = Flask(__name__)

logger = logging.getLogger(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analysis")
def analysis():
    try:
        with open("ANALYSIS.md", "r", encoding="utf-8") as f:
            analysis_content = f.read()
        return jsonify({"success": True, "data": analysis_content})
    except FileNotFoundError:
        return jsonify({"success": False, "message": "Analysis not available"}), 404


@app.route("/history")
def history():
    try:
        with open("HISTORY.md", "r", encoding="utf-8") as f:
            history_content = f.read()
        return jsonify({"success": True, "data": history_content})
    except FileNotFoundError:
        return jsonify({"success": False, "message": "History not available"}), 404


def run_server():
    logger.info("Starting Flask server...")
    app.run(debug=True, use_reloader=False)


if __name__ == "__main__":
    # Cleanup Flask server before exit
    def cleanup(signum, frame):
        logger.info("Shutting down Flask server...")
        os._exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    run_server()
