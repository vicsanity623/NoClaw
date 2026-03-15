import json
from typing import Any


class CascadeQueueHandler:
    def __init__(self, controller: Any):
        self.controller = controller

    def handle_add_to_cascade_queue(self, item: str):
        try:
            self.controller.add_to_cascade_queue(item)
            return json.dumps(
                {"message": f"Item '{item}' added to cascade queue successfully"}
            ).encode()
        except AttributeError:
            return json.dumps(
                {
                    "error": "Controller method 'add_to_cascade_queue' not found. Ensure entrance.py is updated."
                }
            ).encode()
        except Exception as e:
            return json.dumps({"error": f"Internal server error: {str(e)}"}).encode()

    def handle_remove_from_cascade_queue(self, item_id: str):
        try:
            self.controller.remove_cascade_queue_item(item_id)
            return json.dumps(
                {"message": f"Item {item_id} removed successfully"}
            ).encode()
        except AttributeError:
            return json.dumps(
                {
                    "error": "Controller method 'remove_cascade_queue_item' not found. Ensure entrance.py is updated."
                }
            ).encode()
        except Exception as e:
            return json.dumps({"error": f"Internal server error: {str(e)}"}).encode()

    def handle_move_cascade_queue_item(self, item_id: str, direction: str):
        try:
            self.controller.move_cascade_queue_item(item_id, direction)
            return json.dumps(
                {"message": f"Item {item_id} moved {direction} successfully"}
            ).encode()
        except AttributeError:
            return json.dumps(
                {
                    "error": "Controller method 'move_cascade_queue_item' not found. Ensure entrance.py is updated."
                }
            ).encode()
        except Exception as e:
            return json.dumps({"error": f"Internal server error: {str(e)}"}).encode()
