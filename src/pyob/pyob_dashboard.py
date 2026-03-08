import json
import os
from http.server import BaseHTTPRequestHandler
from typing import Any

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
        /* New styles for patch review */
        .patch-card { background: rgba(0, 255, 65, 0.1); border: 1px solid #008F11; padding: 10px; margin-bottom: 10px; }
        .patch-content { font-family: 'Courier New', monospace; font-size: 0.8em; background: #000; border: 1px solid #004411; padding: 5px; margin-top: 5px; max-height: 200px; overflow-y: auto; }
        .patch-actions button { padding: 5px 10px; margin-right: 5px; cursor: pointer; border: none; }
        .approve-btn { background: #00FF41; color: #000; }
        .reject-btn { background: #FF0000; color: #fff; }
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
    <!-- New section for Pending Architectural Patches -->
    <div class="border card">
        <div class="label">PENDING ARCHITECTURAL PATCHES:</div>
        <div id="pendingPatches" class="data" style="height: auto; min-height: 150px;">
            <!-- Patches will be loaded here -->
            No pending patches.
        </div>
    </div>
    <div class="border card">
        <div class="label">MANUAL TARGET OVERRIDE:</div>
        <input type="text" id="manualTargetFile" placeholder="e.g., src/pyob/new_feature.py" style="width: calc(100% - 120px); padding: 8px; margin-right: 10px; background: #000; border: 1px solid #00FF41; color: #00FF41;">
        <button onclick="setManualTarget()" style="padding: 8px 15px; background: #00FF41; color: #000; border: none; cursor: pointer;">Set Next Target</button>
        <div id="targetMessage" style="margin-top: 10px; font-size: 0.9em;"></div>
    </div>
    <script>
        async function updateStats() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                document.getElementById('iteration').innerText = data.iteration || "0";
                document.getElementById('ledger').innerText = data.ledger_stats?.definitions || "0";
                const patchCount = data.patches_count || 0;
                document.getElementById('status').innerText = (data.cascade_queue?.length > 0 || patchCount > 0) ? "EVOLVING" : "READY";
                document.getElementById('memory').innerText = data.memory || "Initializing brain...";
                document.getElementById('history').innerText = data.history || "No history recorded yet.";
                document.getElementById('analysis').innerText = data.analysis || "Parsing directory structure...";
                const queueDiv = document.getElementById('queue');
                queueDiv.innerHTML = data.cascade_queue?.length > 0
                    ? data.cascade_queue.map(f => `<span class='queue-item'>${f}</span>`).join('')
                    : "IDLE // NO PENDING CASCADES";

                if (patchCount > 0) {
                    const pResponse = await fetch('/api/pending_patches');
                    const pData = await pResponse.json();
                    renderPatches(pData.patches);
                } else {
                    renderPatches([]); // Clear the list if no patches
                }
            } catch (e) {
                console.error("Error updating stats:", e);
                document.getElementById('status').innerText = "OFFLINE";
            }
        }

        function renderPatches(patches = []) { // Added default empty array
            const patchesDiv = document.getElementById('pendingPatches');
            if (!patches || patches.length === 0) { // Added null check
                patchesDiv.innerHTML = "No pending patches.";
                return;
            }
            patchesDiv.innerHTML = patches.map(patch => `
                <div class="patch-card">
                    <div class="label">Target: ${patch.file}</div>
                    <pre class="patch-content">${patch.explanation}</pre>
                    <div class="patch-actions" style="margin-top: 10px;">
                        <div style="color: #888; font-size: 0.8em; margin-bottom: 5px;">Review on GitHub PR to Apply</div>
                    </div>
                </div>
            `).join('');
        }

        async function setManualTarget() {
            const targetFile = document.getElementById('manualTargetFile').value;
            const targetMessageDiv = document.getElementById('targetMessage');
            if (!targetFile) {
                targetMessageDiv.innerText = "Please enter a file path.";
                targetMessageDiv.style.color = 'red';
                return;
            }
            try {
                const response = await fetch('/api/set_target_file', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target_file: targetFile })
                });
                const data = await response.json();
                if (response.ok) {
                    targetMessageDiv.innerText = `Target set: ${data.target_file}`;
                    targetMessageDiv.style.color = '#00FF41';
                    document.getElementById('manualTargetFile').value = ''; // Clear input
                } else {
                    targetMessageDiv.innerText = `Error: ${data.error || 'Failed to set target.'}`;
                    targetMessageDiv.style.color = 'red';
                }
            } catch (e) {
                targetMessageDiv.innerText = `Network error: ${e.message}`;
                targetMessageDiv.style.color = 'red';
            }
        }

        function renderPatches(patches) {
            const patchesDiv = document.getElementById('pendingPatches');
            if (patches.length === 0) {
                patchesDiv.innerHTML = "No pending patches.";
                return;
            }
            patchesDiv.innerHTML = patches.map(patch => `
                <div class="patch-card">
                    <div class="label">Patch ID: ${patch.id}</div>
                    <pre class="patch-content">${patch.content}</pre>
                    <div class="patch-actions" style="margin-top: 10px;">
                        <button class="approve-btn" onclick="reviewPatch('${patch.id}', 'approve')">Approve</button>
                        <button class="reject-btn" onclick="reviewPatch('${patch.id}', 'reject')">Reject</button>
                    </div>
                </div>
            `).join('');
        }

        async function reviewPatch(patchId, action) {
            try {
                const response = await fetch('/api/review_patch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ patch_id: patchId, action: action })
                });
                const data = await response.json();
                if (response.ok) {
                    alert(`Patch ${patchId} ${action}d successfully.`);
                    updateStats(); // Refresh dashboard to reflect changes
                } else {
                    alert(`Error ${action}ing patch ${patchId}: ${data.error || 'Unknown error'}`);
                }
            } catch (e) {
                alert(`Network error while ${action}ing patch ${patchId}: ${e.message}`);
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
                self.send_response(503)
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
            pr_content = self.controller._read_file(
                os.path.join(self.controller.pyob_dir, "PEER_REVIEW.md")
            )
            patch_count = pr_content.count("## 🛠 Review for")

            status = {
                "iteration": getattr(self.controller, "current_iteration", 1),
                "cascade_queue": self.controller.cascade_queue,
                "patches_count": patch_count,
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
                pr_path = os.path.join(self.controller.pyob_dir, "PEER_REVIEW.md")
                patches = []
                if os.path.exists(pr_path):
                    content = self.controller._read_file(pr_path)
                    patches.append(
                        {
                            "file": "PEER_REVIEW.md",
                            "explanation": content[:500] + "...",
                            "id": "PR_1",
                        }
                    )

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"patches": patches}).encode())
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
                action = data.get("action")

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

                self.controller.process_patch_review(patch_id, action)
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

    def log_message(self, format: str, *args: Any) -> None:
        return
