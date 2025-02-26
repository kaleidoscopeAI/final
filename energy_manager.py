import logging
from collections import defaultdict

# Configure logging for EnergyManager
logging.basicConfig(
    filename="energy_manager.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class EnergyManager:
    def __init__(self, total_energy=1000.0):
        """
        Manages energy distribution and consumption for nodes and supernodes.

        Args:
            total_energy (float): Total energy available to the system.
        """
        self.total_energy = total_energy
        self.node_energy = defaultdict(float)
        self.supernode_energy = defaultdict(float)

    def allocate_node_energy(self, node_id, energy_amount):
        """
        Allocates energy to a specific node.

        Args:
            node_id (str): Unique identifier for the node.
            energy_amount (float): Amount of energy to allocate.

        Returns:
            None
        """
        if self.total_energy >= energy_amount:
            self.node_energy[node_id] += energy_amount
            self.total_energy -= energy_amount
            logging.info(f"Allocated {energy_amount} energy to node {node_id}")
        else:
            logging.warning(f"Insufficient energy to allocate to node {node_id}")

    def allocate_supernode_energy(self, supernode_id, energy_amount):
        """
        Allocates energy to a specific supernode.

        Args:
            supernode_id (str): Unique identifier for the supernode.
            energy_amount (float): Amount of energy to allocate.

        Returns:
            None
        """
        if self.total_energy >= energy_amount:
            self.supernode_energy[supernode_id] += energy_amount
            self.total_energy -= energy_amount
            logging.info(f"Allocated {energy_amount} energy to supernode {supernode_id}")
        else:
            logging.warning(f"Insufficient energy to allocate to supernode {supernode_id}")

    def consume_node_energy(self, node_id, energy_consumed):
        """
        Consumes energy from a specific node.

        Args:
            node_id (str): Unique identifier for the node.
            energy_consumed (float): Amount of energy consumed.

        Returns:
            None
        """
        if self.node_energy[node_id] >= energy_consumed:
            self.node_energy[node_id] -= energy_consumed
            logging.info(f"Node {node_id} consumed {energy_consumed} energy")
        else:
            logging.warning(f"Node {node_id} has insufficient energy")

    def consume_supernode_energy(self, supernode_id, energy_consumed):
        """
        Consumes energy from a specific supernode.

        Args:
            supernode_id (str): Unique identifier for the supernode.
            energy_consumed (float): Amount of energy consumed.

        Returns:
            None
        """
        if self.supernode_energy[supernode_id] >= energy_consumed:
            self.supernode_energy[supernode_id] -= energy_consumed
            logging.info(f"Supernode {supernode_id} consumed {energy_consumed} energy")
        else:
            logging.warning(f"Supernode {supernode_id} has insufficient energy")

    def get_remaining_energy(self):
        """
        Returns the total remaining energy in the system.

        Returns:
            float: Remaining energy.
        """
        return self.total_energy

# Example usage
if __name__ == "__main__":
    energy_manager = EnergyManager(total_energy=1000)
    energy_manager.allocate_node_energy("node1", 100)
    energy_manager.allocate_supernode_energy("supernode1", 200)
    energy_manager.consume_node_energy("node1", 50)
    energy_manager.consume_supernode_energy("supernode1", 100)
    remaining_energy = energy_manager.get_remaining_energy()
    print(f"Remaining energy: {remaining_energy}")
