import threading
import time

class NetworkManager:
    def __init__(self):
        self.nodes = {}

    def monitor(self):
        while True:
            for node_id, status in self.nodes.items():
                print(f"Node {node_id}: {status}")
            time.sleep(5)

    def get_status(self):
        return {"nodes_count": len(self.nodes)}
