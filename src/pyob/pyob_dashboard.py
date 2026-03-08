import json
import os
from http.server import BaseHTTPRequestHandler
from typing import Any, Optional

OBSERVER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>PyOB // OBSERVER</title>
    <style>
        body { background: #050505; color: #00FF41; font-family: 'Menlo', monospace; margin: 0; padding: 20px; overflow-x: hidden; }
        .glow { text-shadow: 0 0 10px #00FF41, 0 0 20px #00FF41; }
        .border { border: 1px solid #00FF41; box-shadow: 0 0 15px rgba(0, 255, 65, 0.2); padding: 20px; margin-bottom: 20px; }
        h1 { font-size: 2em; border-bottom: 2px solid #00FF41; padding-bottom: 10px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background: rgba(0, 255, 65, 0.05); }
        .label { color: #008F11; font-weight: bold; margin-bottom: 5px; }
        .data { font-size: 0.9em; white-space: pre-wrap; height: 300px; overflow-y: auto; border: 1px solid #004411; padding: 10px; background: #000; }
        .stat-bar { display: flex; justify-content: space-between; font-size: 1.2em; margin-bottom: 20px; }
        .queue-item { background: #00FF41; color: #000; padding: 2px 5px; margin: 2px; display: inline-block; font-size: 0.8em; }
        #iteration { font-size: 1.5em; color: #fff; }
    </style>
</head>
<body>
    <h1 class="glow">PYOB_OS // OBSERVER_DASHBOARD</h1>
    <div class="stat-bar border card">
        <div>ITERATION: <span id="iteration" class="glow">--</span></div>
        <div>LEDGER: <span id="ledger">--</span> symbols</div>
        <div>STATUS: <span id="status" style="color: #fff">SCANNING...</span></div>
    </div>
    <div class="border card"><div class="label">SYMBOLIC CASCADE QUEUE:</div><div id="queue">--</div></div>
    <div class="grid">
        <div class="border card"><div class="label">LIVE MEMORY (MEMORY.md):</div><div id="memory" class="data">--</div></div>
        <div class="border card"><div class="label">RECENT HISTORY (HISTORY.md):</div><div id="history" class="data">--</div></div>
    </div>
    <div class="border card"><div class="label">LATEST ARCHITECTURAL ANALYSIS:</div><div id="analysis" class="data">--</div></div>
    <script>
        async function updateStats() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('iteration').innerText = data.iteration;
                document.getElementById('ledger').innerText = data.ledger_stats.definitions;
                document.getElementById('memory').innerText = data.memory || "Initializing brain...";
                document.getElementById('history').innerText = data.history || "No history recorded yet.";
                document.getElementById('analysis').innerText = data.analysis || "Parsing directory structure...";
                const queueDiv = document.getElementById('queue');
                queueDiv.innerHTML = data.cascade_queue.length > 0 ? data.cascade_queue.map(f => `<span class='queue-item'>${f}</span>`).join('') : "IDLE // NO PENDING CASCADES";
                document.getElementById('status').innerText = data.cascade_queue.length > 0 ? "EVOLVING" : "READY";
            } catch (e) { document.getElementById('status').innerText = "OFFLINE"; }
        }
        setInterval(updateStats, 3000);
        updateStats();
    </script>
</body>
</html>
"""


class ObserverHandler(BaseHTTPRequestHandler):
    controller: Optional[Any] = None

    def do_GET(self):
        if self.path == "/api/status":
            if self.controller is None:
                self.send_response(503)  # Service Unavailable
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": "Controller not initialized"}).encode()
                )
                return
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            status = {
                "iteration": getattr(self.controller, "current_iteration", 1),
                "cascade_queue": self.controller.cascade_queue,
                "ledger_stats": {
                    "definitions": len(self.controller.ledger["definitions"]),
                    "references": len(self.controller.ledger["references"]),
                },
                "analysis": self.controller._read_file(self.controller.analysis_path),
                "memory": self.controller._read_file(
                    os.path.join(self.controller.target_dir, ".pyob", "MEMORY.md")
                ),
                "history": self.controller._read_file(self.controller.history_path)[
                    -5000:
                ],
            }
            self.wfile.write(json.dumps(status).encode())
        elif self.path == "/" or self.path == "/observer.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(OBSERVER_HTML.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return
