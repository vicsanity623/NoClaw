import json
import os
from http.server import BaseHTTPRequestHandler
from typing import Any

OBSERVER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyOB // ARCHITECT HUD</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;600&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #0a0a0c; --card: #141417; --accent: #00ffa3; --text: #e0e0e6; --dim: #88888e; --err: #ff4d4d; }
        * { box-sizing: border-box; }
        body { background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; margin: 0; padding: 15px; line-height: 1.5; }
        .hud-container { max-width: 1200px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        /* Typography & Glow */
        h1 { grid-column: span 2; font-family: 'JetBrains Mono'; font-size: 1.2rem; letter-spacing: 2px; color: var(--accent); text-transform: uppercase; margin: 10px 0; display: flex; justify-content: space-between; }
        .glow { text-shadow: 0 0 15px var(--accent); }
        /* Component Cards */
        .card { background: var(--card); border: 1px solid #2a2a30; border-radius: 8px; padding: 20px; overflow: hidden; position: relative; }
        .card::before { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 2px; background: linear-gradient(90deg, transparent, var(--accent), transparent); opacity: 0.3; }
        .label { font-size: 0.7rem; font-weight: 600; color: var(--dim); text-transform: uppercase; margin-bottom: 12px; letter-spacing: 1px; display: flex; align-items: center; gap: 8px; }
        .label::before { content: ''; width: 6px; height: 6px; background: var(--accent); border-radius: 50%; box-shadow: 0 0 8px var(--accent); }
        /* Data Displays */
        .data-box { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; height: 250px; overflow-y: auto; background: #00000044; border-radius: 4px; padding: 12px; color: #ced4e0; scrollbar-width: thin; }
        .stat-grid { grid-column: span 2; display: flex; gap: 40px; background: var(--card); padding: 15px 25px; border-radius: 8px; border: 1px solid #2a2a30; }
        .stat-item { display: flex; flex-direction: column; }
        .stat-val { font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono'; color: #fff; }
        .stat-lbl { font-size: 0.6rem; color: var(--dim); }
        /* Mobile Specifics */
        @media (max-width: 768px) {
            .hud-container { grid-template-columns: 1fr; }
            h1, .stat-grid { grid-column: 1; }
            .stat-grid { flex-wrap: wrap; gap: 20px; }
        }
        .status-pill { padding: 4px 12px; border-radius: 20px; font-size: 0.7rem; font-weight: 800; background: #222; }
        .evolving { color: var(--accent); border: 1px solid var(--accent); box-shadow: 0 0 10px #00ffa344; }
        input { background: #000; border: 1px solid #2a2a30; color: var(--accent); padding: 10px; border-radius: 4px; width: 100%; font-family: 'JetBrains Mono'; margin-bottom: 10px; }
        button { width: 100%; padding: 12px; background: var(--accent); color: #000; border: none; border-radius: 4px; font-weight: 700; cursor: pointer; transition: 0.2s; }
        button:hover { filter: brightness(1.2); }
    </style>
</head>
<body>
    <h1>
        <span>PyOB // Evolution Engine</span>
        <span id="status-pill" class="status-pill">READY</span>
    </h1>
    <div class="hud-container">
        <div class="stat-grid">
            <div class="stat-item"><span class="stat-lbl">Iteration</span><span id="iteration" class="stat-val">--</span></div>
            <div class="stat-item"><span class="stat-lbl">Symbolic Ledger</span><span id="ledger" class="stat-val">--</span></div>
            <div class="stat-item"><span class="stat-lbl">Pending Cascades</span><span id="queue-count" class="stat-val">--</span></div>
        </div>
        <div class="card">
            <div class="label">Logic Memory (MEMORY.md)</div>
            <div id="memory" class="data-box">Initializing brain...</div>
        </div>
        <div class="card">
            <div class="label">System Logs (HISTORY.md)</div>
            <div id="history" class="data-box">No history yet.</div>
        </div>
        <div class="card" style="grid-column: span 2;">
            <div class="label">Architectural Analysis</div>
            <div id="analysis" class="data-box" style="height: 350px;">Scanning structure...</div>
        </div>
        <div class="card" style="grid-column: span 2;">
            <div class="label">Pending Patch Reviews</div>
            <div id="pending-patches" class="data-box" style="height: 250px;">No pending patches.</div>
        </div>
        <div class="card">
            <div class="label">Manual Override</div>
            <input type="text" id="manualTargetFile" placeholder="src/pyob/target.py">
            <button onclick="setManualTarget()">FORCE TARGET</button>
        </div>
        <div class="card">
            <div class="label">Queue Status</div>
            <div id="queue" class="data-box" style="height: 100px;">IDLE</div>
        </div>
    </div>

    <script>
        async function updateStats() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('iteration').innerText = data.iteration || "0";
                document.getElementById('ledger').innerText = (data.ledger_stats?.definitions || 0) + " SYM";
                document.getElementById('queue-count').innerText = data.cascade_queue?.length || "0";
                const pill = document.getElementById('status-pill');
                const isEvolving = data.cascade_queue?.length > 0 || data.patches_count > 0;
                pill.innerText = isEvolving ? "EVOLVING" : "STABLE";
                pill.className = isEvolving ? "status-pill evolving" : "status-pill";
                document.getElementById('memory').innerText = data.memory || "Brain empty.";
                document.getElementById('history').innerText = data.history || "No logs.";
                document.getElementById('analysis').innerText = data.analysis || "Parsing...";
                const queueDiv = document.getElementById('queue');
                queueDiv.innerText = data.cascade_queue?.length > 0 ? data.cascade_queue.join('\\n') : "EMPTY";
                await updatePendingPatches(); // Refresh pending patches
            } catch (e) { document.getElementById('status-pill').innerText = "OFFLINE"; }
        }

        async function setManualTarget() {
            const targetFile = document.getElementById('manualTargetFile').value;
            if (!targetFile) return;
            await fetch('/api/set_target_file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target_file: targetFile })
            });
            document.getElementById('manualTargetFile').value = '';
        }

        async function updatePendingPatches() {
            try {
                const response = await fetch('/api/pending_patches');
                const data = await response.json();
                const patchesDiv = document.getElementById('pending-patches');
                patchesDiv.innerHTML = ''; // Clear previous content

                if (data.patches && data.patches.length > 0) {
                    data.patches.forEach(patch => {
                        const patchElement = document.createElement('div');
                        patchElement.style.marginBottom = '10px';
                        patchElement.innerHTML = `
                            <span style="color: var(--accent); font-weight: 700;">Patch ID: ${patch.id}</span><br>
                            <span style="font-size: 0.8em; color: var(--dim);">File: ${patch.target_file}</span><br>
                            <span style="font-size: 0.8em; color: var(--dim);">Description: ${patch.description || 'N/A'}</span><br>
                            <button onclick="reviewPatch('${patch.id}', 'approve')" style="width: 48%; margin-right: 4%; background: #00cc66; color: #000;">APPROVE</button>
                            <button onclick="reviewPatch('${patch.id}', 'reject')" style="width: 48%; background: #cc0000; color: #fff;">REJECT</button>
                        `;
                        patchesDiv.appendChild(patchElement);
                    });
                } else {
                    patchesDiv.innerText = "No pending patches.";
                }
            } catch (e) {
                console.error("Failed to fetch pending patches:", e);
                document.getElementById('pending-patches').innerText = "Error loading patches.";
            }
        }

        async function reviewPatch(patchId, action) {
            try {
                await fetch('/api/review_patch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ patch_id: patchId, action: action })
                });
                // Refresh stats and patches after review
                await updateStats();
            } catch (e) {
                console.error(`Failed to ${action} patch ${patchId}:`, e);
                alert(`Failed to ${action} patch ${patchId}. Check console for details.`);
            }
        }

        setInterval(updateStats, 3000);
        updateStats();
    </script>
</body>
</html>
"""


class ObserverHandler(BaseHTTPRequestHandler):
    controller: Any = None

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
                "patches_count": len(self.controller.get_pending_patches())
                if hasattr(self.controller, "get_pending_patches")
                else 0,
            }
            self.wfile.write(json.dumps(status).encode())
        # New GET endpoint for pending patches
        elif self.path == "/api/pending_patches":
            if self.controller is None:
                self.send_response(503)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": "Controller not initialized"}).encode()
                )
                return
            try:
                pending_patches = (
                    self.controller.get_pending_patches()
                )  # Assumes this method exists in EntranceController
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"patches": pending_patches}).encode())
            except AttributeError:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "error": "Controller method 'get_pending_patches' not found. Ensure entrance.py is updated."
                        }
                    ).encode()
                )
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": f"Internal server error: {str(e)}"}).encode()
                )
        elif self.path == "/" or self.path == "/observer.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(OBSERVER_HTML.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/set_target_file":
            if self.controller is None:
                self.send_response(503)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": "Controller not initialized"}).encode()
                )
                return

            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                target_file = data.get("target_file")

                if not target_file:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {"error": "Missing 'target_file' in request body"}
                        ).encode()
                    )
                    return

                # This method call depends on entrance.py being updated
                self.controller.set_manual_target_file(target_file)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "message": "Manual target file set",
                            "target_file": target_file,
                        }
                    ).encode()
                )

            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            except AttributeError:
                # If controller doesn't have set_manual_target_file yet
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "error": "Controller method 'set_manual_target_file' not found. Ensure entrance.py is updated."
                        }
                    ).encode()
                )
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": f"Internal server error: {str(e)}"}).encode()
                )
        # New POST endpoint for reviewing patches
        elif self.path == "/api/review_patch":
            if self.controller is None:
                self.send_response(503)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": "Controller not initialized"}).encode()
                )
                return

            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                patch_id = data.get("patch_id")
                action = data.get("action")  # 'approve' or 'reject'

                if not patch_id or not action:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {"error": "Missing 'patch_id' or 'action' in request body"}
                        ).encode()
                    )
                    return
                if action not in ["approve", "reject"]:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {"error": "Action must be 'approve' or 'reject'"}
                        ).encode()
                    )
                    return

                self.controller.process_patch_review(
                    patch_id, action
                )  # Assumes this method exists in EntranceController

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "message": f"Patch {patch_id} {action}d successfully",
                            "patch_id": patch_id,
                            "action": action,
                        }
                    ).encode()
                )

            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            except AttributeError:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "error": "Controller method 'process_patch_review' not found. Ensure entrance.py is updated."
                        }
                    ).encode()
                )
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": f"Internal server error: {str(e)}"}).encode()
                )
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return
