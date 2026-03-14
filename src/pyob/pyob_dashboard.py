import json
import os
from http.server import BaseHTTPRequestHandler
from typing import Any

from .dashboard_html import OBSERVER_HTML


class ObserverHandler(BaseHTTPRequestHandler):
    # The 'controller' type is 'Any' to avoid circular dependencies with the main application controller.
    controller: Any = None

    def _send_controller_not_initialized_error(self):
        self.send_response(503)  # Service Unavailable
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Controller not initialized"}).encode())

    def do_GET(self):
        if self.path == "/api/status":
            if self.controller is None:
                self._send_controller_not_initialized_error()
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
                self._send_controller_not_initialized_error()
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
        # NEW POST endpoint for updating Logic Memory
        elif self.path == "/api/update_memory":
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
                new_memory_content = data.get("content")

                if new_memory_content is None:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {"error": "Missing 'content' in request body"}
                        ).encode()
                    )
                    return

                self.controller.update_memory(new_memory_content)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"message": "Logic Memory updated successfully"}
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
                            "error": "Controller method 'update_memory' not found. Ensure entrance.py is updated."
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
        # NEW POST endpoint for moving cascade queue items
        elif self.path == "/api/cascade_queue/move":
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
                item_id = data.get("item_id")
                direction = data.get("direction")

                if not item_id or direction not in ["up", "down"]:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "error": "Missing 'item_id' or invalid 'direction' in request body"
                            }
                        ).encode()
                    )
                    return

                self.controller.move_cascade_queue_item(item_id, direction)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"message": f"Item {item_id} moved {direction} successfully"}
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
                            "error": "Controller method 'move_cascade_queue_item' not found. Ensure entrance.py is updated."
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

        # NEW POST endpoint for removing cascade queue items
        elif self.path == "/api/cascade_queue/remove":
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
                item_id = data.get("item_id")

                if not item_id:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {"error": "Missing 'item_id' in request body"}
                        ).encode()
                    )
                    return

                self.controller.remove_cascade_queue_item(item_id)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"message": f"Item {item_id} removed successfully"}
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
                            "error": "Controller method 'remove_cascade_queue_item' not found. Ensure entrance.py is updated."
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
        # NEW POST endpoint for adding items to cascade queue
        elif self.path == "/api/cascade_queue/add":
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
                item = data.get("item")

                if not item:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"error": "Missing 'item' in request body"}).encode()
                    )
                    return

                self.controller.add_to_cascade_queue(item)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "message": f"Item '{item}' added to cascade queue successfully"
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
                            "error": "Controller method 'add_to_cascade_queue' not found. Ensure entrance.py is updated."
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
