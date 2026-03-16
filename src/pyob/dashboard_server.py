import json
import logging
import os
import signal
import sys
from datetime import datetime

from flask import Flask, jsonify, render_template

from pyob.data_parser import DataParser

app = Flask(__name__)

logger = logging.getLogger(__name__)
data_parser_instance = DataParser()  # Initialize DataParser once globally


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
    except UnicodeDecodeError:
        return jsonify(
            {
                "success": False,
                "message": "Error reading history content due to encoding issue",
            }
        ), 500


@app.route("/api/analysis/issues/<string:issue_id>/acknowledge", methods=["POST"])
def acknowledge_issue(issue_id):
    """
    API endpoint to acknowledge a specific analysis issue.
    The status is stored in a simple JSON file.
    """
    try:
        status_file = "issue_statuses.json"
        issue_statuses = {}
        if os.path.exists(status_file):
            with open(status_file, "r", encoding="utf-8") as f:
                issue_statuses = json.load(f)

        # Update status for the given issue_id
        issue_statuses[issue_id] = {
            "status": "acknowledged",
            "timestamp": datetime.now().isoformat(),
        }

        # Save updated statuses
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(issue_statuses, f, indent=4)

        logger.info(f"Issue {issue_id} acknowledged by user.")
        return jsonify({"success": True, "message": f"Issue {issue_id} acknowledged."})
    except Exception as e:
        logger.error(f"Error acknowledging issue {issue_id}: {e}")
        return jsonify(
            {"success": False, "message": "Failed to acknowledge issue."}
        ), 500


@app.route("/api/analysis-data")
def api_analysis_data():
    """
    Returns parsed analysis data, enriched with acknowledgment statuses.
    Assumes DataParser provides unique 'id' for each issue.
    """
    try:
        analysis_content = read_file("ANALYSIS.md")
        data_parser = DataParser()
        parsed_data = data_parser.parse_analysis_content(analysis_content)

        issue_statuses = {}
        status_file = "issue_statuses.json"
        if os.path.exists(status_file):
            with open(status_file, "r", encoding="utf-8") as f:
                issue_statuses = json.load(f)

        # Merge statuses into parsed_data.
        # This logic assumes parsed_data is a dictionary with an 'issues' key,
        # where 'issues' is a list of dictionaries, each with an 'id'.
        # Adjust if DataParser returns a different structure.
        if isinstance(parsed_data, dict) and "issues" in parsed_data:
            for issue in parsed_data.get("issues", []):
                issue_id = issue.get("id")  # DataParser must provide unique IDs
                if issue_id and issue_id in issue_statuses:
                    issue["status"] = issue_statuses[issue_id]["status"]
                    issue["acknowledged_at"] = issue_statuses[issue_id]["timestamp"]
                else:
                    issue["status"] = "new"  # Default status for unacknowledged issues
        elif isinstance(
            parsed_data, list
        ):  # Fallback if DataParser returns a list directly
            for issue in parsed_data:
                issue_id = issue.get("id")
                if issue_id and issue_id in issue_statuses:
                    issue["status"] = issue_statuses[issue_id]["status"]
                    issue["acknowledged_at"] = issue_statuses[issue_id]["timestamp"]
                else:
                    issue["status"] = "new"

        return jsonify(parsed_data)
    except FileNotFoundError:
        return jsonify(
            {"success": False, "message": "Analysis data not available"}
        ), 404
    except UnicodeDecodeError:
        return jsonify(
            {
                "success": False,
                "message": "Error parsing analysis content due to encoding issue",
            }
        ), 500


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
    except FileNotFoundError as e:
        logger.error(f"File not found: {filename}: {e}")
        raise
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
