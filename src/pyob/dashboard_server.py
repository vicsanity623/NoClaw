import logging
import os
import signal
import sys

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


@app.route("/api/analysis-data")
def api_analysis_data():
    try:
        analysis_content = read_file("ANALYSIS.md")
        parsed_data = parse_analysis_content(analysis_content)
        return jsonify(parsed_data)
    except FileNotFoundError:
        return jsonify(
            {"success": False, "message": "Analysis data not available"}
        ), 404


@app.route("/api/history-data")
def api_history_data():
    try:
        history_content = read_file("HISTORY.md")
        parsed_data = parse_history_content(history_content)
        return jsonify(parsed_data)
    except FileNotFoundError:
        return jsonify({"success": False, "message": "History data not available"}), 404


def read_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def parse_analysis_content(content):
    # Implement logic to parse the analysis content into structured data
    return {
        "data": [{"key": "example1", "value": 10}, {"key": "example2", "value": 20}]
    }


def parse_history_content(content):
    # Implement logic to parse the history content into structured data
    return {
        "data": [
            {"event": "event1", "date": "2022-01-01"},
            {"event": "event2", "date": "2023-01-01"},
        ]
    }


def run_server():
    logger.info("Starting Flask server...")
    # Use an environment variable to control debug mode for safety
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode, use_reloader=False)


if __name__ == "__main__":
    # Cleanup Flask server before exit
    def cleanup(signum, frame):
        logger.info("Shutting down Flask server...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    run_server()
