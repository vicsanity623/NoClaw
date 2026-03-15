import json

from dashboard_server import fetch_api


class StatsUpdater:
    async def update_stats(self):
        try:
            response = await fetch_api("/api/status")
            data = await response.json()
            return data
        except Exception as e:
            print(f"Error updating stats: {e}")
            return None

    async def update_pending_patches(self):
        try:
            response = await fetch_api("/api/pending_patches")
            data = await response.json()
            return data
        except Exception as e:
            print(f"Failed to fetch pending patches: {e}")
            return None

    async def review_patch(self, patch_id, action):
        try:
            await fetch_api(
                "/api/review_patch",
                method="POST",
                data=json.dumps({"patch_id": patch_id, "action": action}),
            )
        except Exception as e:
            print(f"Failed to {action} patch {patch_id}: {e}")

    async def save_memory(self, memory_content):
        try:
            await fetch_api(
                "/api/update_memory",
                method="POST",
                data=json.dumps({"content": memory_content}),
            )
        except Exception as e:
            print(f"Failed to save Logic Memory: {e}")

    async def add_cascade_item(self, item):
        try:
            await fetch_api(
                "/api/cascade_queue/add", method="POST", data=json.dumps({"item": item})
            )
        except Exception as e:
            print(f"Failed to add item to cascade queue: {e}")

    async def move_queue_item(self, item_id, direction):
        try:
            await fetch_api(
                "/api/cascade_queue/move",
                method="POST",
                data=json.dumps({"item_id": item_id, "direction": direction}),
            )
        except Exception as e:
            print(f"Failed to move item {item_id} {direction}: {e}")

    async def remove_queue_item(self, item_id):
        try:
            await fetch_api(
                "/api/cascade_queue/remove",
                method="POST",
                data=json.dumps({"item_id": item_id}),
            )
        except Exception as e:
            print(f"Failed to remove item {item_id}: {e}")
