import logging
import os
import signal
import sys

from flask import Flask, jsonify, render_template

from pyob.data_parser import DataParser

app = Flask(__name__)

logger = logging.getLogger(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analysis")
def analysis():
    try:
        analysis_content = read_file("ANALYSIS.md")
        return jsonify({"success": True, "data": analysis_content})
    except FileNotFoundError:
        return jsonify({"success": False, "message": "Analysis not available"}), 404
    except UnicodeDecodeError:
        return jsonify(
            {
                "success": False,
                "message": "Error reading analysis content due to encoding issue",
            }
        ), 500


@app.route("/history")
def history():
    try:
        history_content = read_file("HISTORY.md")
        return jsonify({"success": True, "data": history_content})
    except FileNotFoundError:
        return jsonify({"success": False, "message": "History not available"}), 404


@app.route("/api/analysis-data")
def api_analysis_data():
    try:
        analysis_content = read_file("ANALYSIS.md")
        data_parser = DataParser()
        parsed_data = data_parser.parse_analysis_content(analysis_content)
        return jsonify(parsed_data)
    except FileNotFoundError:
        return jsonify(
            {"success": False, "message": "Analysis data not available"}
        ), 404


@app.route("/api/history-data")
def api_history_data():
    try:
        history_content = read_file("HISTORY.md")
        data_parser = DataParser()
        parsed_data = data_parser.parse_history_content(history_content)
        return jsonify(parsed_data)
    except FileNotFoundError:
        return jsonify({"success": False, "message": "History data not available"}), 404


def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError as e:
        logger.error(f"Error reading file {filename}: {e}")
        raise


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
