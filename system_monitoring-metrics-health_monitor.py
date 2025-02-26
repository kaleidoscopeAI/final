import time

class HealthMonitor:
    def __init__(self, node_manager: NodeLifeCycleManager):
        self.node_manager = node_manager
        self.health_status = {}
        self.last_check = time.time()

    def check_health(self):
        """Check the health status of each node."""
        current_time = time.time()
        for node_id, node in self.node_manager.nodes.items():
            # Check heartbeat
            if current_time - node.last_heartbeat > 30:  # Example threshold
                status = "unhealthy"
            else:
                status = "healthy"

            # Check task status
            for task in node.tasks:
                if task.get("status") == "failed":
                    status = "unhealthy"
                    break

            self.health_status[node_id] = {
                'timestamp': current_time,
                'status': status
            }

        self.last_check = current_time

    def get_health_status(self, node_id: str = None) -> dict:
        """Get the latest health status, optionally for a specific node."""
        if node_id:
            return self.health_status.get(node_id, {})
        else:
            return self.health_status

    def get_unhealthy_nodes(self) -> list:
        """Get a list of unhealthy nodes."""
        return [node_id for node_id, status in self.health_status.items() if status['status'] == "unhealthy"]

    def get_state(self) -> dict:
        """Returns the current state of the health monitor."""
        return {
            'last_check': self.last_check,
            'health_status': self.health_status
        }
