import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import deque

@dataclass
class MemoryBank:
    """
    Simulates a gear/memory bank that physically rotates and transforms data.
    """
    position: float = 0.0  # Current rotational position (0-360 degrees)
    capacity: int = 100    # Maximum data capacity
    momentum: float = 0.0  # Current rotational momentum
    energy_level: float = 1.0
    data_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    connected_banks: List['MemoryBank'] = field(default_factory=list)
    
    def __post_init__(self):
        self.data_queue = deque(maxlen=self.capacity)
        self.resonance_field = np.zeros((360,), dtype=float)
    
    def add_data(self, data_point: Any) -> bool:
        """Add data and return True if rotation occurs."""
        # Convert data to numerical representation
        data_vector = self._vectorize_data(data_point)
        
        # Add to queue and matrix
        self.data_queue.append(data_point)
        if len(self.data_matrix) == 0:
            self.data_matrix = data_vector.reshape(1, -1)
        else:
            self.data_matrix = np.vstack([self.data_matrix, data_vector])
        
        # Check for rotation
        if len(self.data_queue) >= self.capacity:
            self._rotate()
            return True
        return False
    
    def _vectorize_data(self, data_point: Any) -> np.ndarray:
        """Convert any data type to numerical vector."""
        if isinstance(data_point, (int, float)):
            return np.array([data_point])
        elif isinstance(data_point, dict):
            # Extract numerical values and encode strings
            vector = []
            for k, v in sorted(data_point.items()):
                if isinstance(v, (int, float)):
                    vector.append(v)
                elif isinstance(v, str):
                    # Simple hash for string values
                    vector.append(hash(v) % 100 / 100)
            return np.array(vector)
        elif isinstance(data_point, str):
            return np.array([hash(data_point) % 100 / 100])
        return np.array([0.0])

    def _rotate(self) -> float:
        """
        Execute a rotation based on data patterns and energy.
        Returns rotation angle.
        """
        # Calculate rotation force from data patterns
        pattern_force = self._calculate_pattern_force()
        
        # Apply momentum and energy
        total_force = (pattern_force + self.momentum) * self.energy_level
        
        # Update position with smooth rotation
        old_position = self.position
        self.position = (self.position + total_force) % 360
        
        # Update momentum with decay
        self.momentum = total_force * 0.3
        
        # Update resonance field
        self._update_resonance()
        
        # Affect connected banks
        self._propagate_rotation(total_force)
        
        # Clear processed data
        self.data_matrix = np.array([])
        self.data_queue.clear()
        
        return self.position - old_position

    def _calculate_pattern_force(self) -> float:
        """Calculate rotational force from data patterns."""
        if len(self.data_matrix) == 0:
            return 0.0
            
        # Calculate pattern complexity
        eigenvalues = np.linalg.eigvals(self.data_matrix.T @ self.data_matrix)
        pattern_complexity = np.sum(np.abs(eigenvalues))
        
        # Normalize and scale force
        base_force = np.clip(pattern_complexity / self.capacity, 0, 45)
        
        # Add resonance influence
        resonance_factor = self.resonance_field[int(self.position)]
        
        return base_force * (1 + resonance_factor)

    def _update_resonance(self):
        """Update the resonance field based on current rotation."""
        # Create resonance wave
        wave = np.sin(np.linspace(0, 2*np.pi, 360) + self.position * np.pi/180)
        
        # Add to resonance field with decay
        self.resonance_field = 0.7 * self.resonance_field + 0.3 * wave
        
        # Normalize
        self.resonance_field = self.resonance_field / np.max(np.abs(self.resonance_field))

    def _propagate_rotation(self, force: float):
        """Propagate rotation force to connected banks."""
        for bank in self.connected_banks:
            transfer_force = force * 0.5 * self.energy_level
            bank.receive_rotation(transfer_force, self)

    def receive_rotation(self, force: float, source: 'MemoryBank'):
        """Receive rotation force from connected bank."""
        # Calculate received force based on resonance
        resonance_match = np.correlate(self.resonance_field, source.resonance_field)[0]
        received_force = force * (0.5 + 0.5 * resonance_match)
        
        # Update position and momentum
        self.position = (self.position + received_force) % 360
        self.momentum += received_force * 0.1

class LogicPath:
    """
    Represents a dynamic path that generates insights based on bank rotations.
    """
    def __init__(self):
        self.position = 0.0
        self.momentum = 0.0
        self.field_strength = 1.0
        self.pattern_memory = deque(maxlen=1000)
        self.resonance_map = np.zeros((360,), dtype=float)
        
    def shift(self, bank_positions: List[float], resonance_fields: List[np.ndarray]) -> List[Dict]:
        """
        Shift position based on bank rotations and generate insights.
        """
        # Calculate new position from bank influences
        position_influence = np.mean(bank_positions)
        resonance_influence = self._calculate_resonance(resonance_fields)
        
        # Update position with momentum and influences
        old_position = self.position
        self.position = (self.position + position_influence + 
                        resonance_influence + self.momentum) % 360
        
        # Update momentum with decay
        self.momentum = (self.momentum + (self.position - old_position)) * 0.3
        
        # Generate insights based on new position
        return self._generate_insights(bank_positions, resonance_fields)
        
    def _calculate_resonance(self, resonance_fields: List[np.ndarray]) -> float:
        """Calculate resonance influence from bank fields."""
        if not resonance_fields:
            return 0.0
            
        # Combine resonance fields
        combined_field = np.mean(resonance_fields, axis=0)
        
        # Update path's resonance map
        self.resonance_map = 0.8 * self.resonance_map + 0.2 * combined_field
        
        # Calculate influence based on current position
        pos_idx = int(self.position)
        field_slice = self.resonance_map[max(0, pos_idx-10):min(360, pos_idx+11)]
        return np.mean(field_slice) * self.field_strength

    def _generate_insights(self, bank_positions: List[float], 
                         resonance_fields: List[np.ndarray]) -> List[Dict]:
        """Generate insights based on current state."""
        insights = []
        
        # Calculate pattern from current state
        pattern = {
            'position': self.position,
            'momentum': self.momentum,
            'field_strength': self.field_strength,
            'resonance': np.mean(self.resonance_map),
            'bank_alignment': np.std(bank_positions)
        }
        self.pattern_memory.append(pattern)
        
        # Generate insights based on pattern memory
        if len(self.pattern_memory) >= 3:
            recent_patterns = list(self.pattern_memory)[-3:]
            
            # Detect pattern shifts
            position_shifts = [p['position'] for p in recent_patterns]
            momentum_shifts = [p['momentum'] for p in recent_patterns]
            
            # Calculate insight strength from pattern stability
            stability = 1.0 - np.std(position_shifts) / 180
            
            # Generate insight
            insight = {
                'strength': stability * self.field_strength,
                'pattern': pattern,
                'shifts': {
                    'position': np.diff(position_shifts).tolist(),
                    'momentum': np.diff(momentum_shifts).tolist()
                },
                'resonance': self.resonance_map[int(self.position)]
            }
            insights.append(insight)
        
        return insights

# Basic usage example
if __name__ == "__main__":
    # Create memory banks
    bank1 = MemoryBank(capacity=50)
    bank2 = MemoryBank(capacity=75)
    bank3 = MemoryBank(capacity=60)
    
    # Connect banks
    bank1.connected_banks = [bank2]
    bank2.connected_banks = [bank1, bank3]
    bank3.connected_banks = [bank2]
    
    # Create logic path
    path = LogicPath()
    
    # Test with sample data
    test_data = [
        {"value": i, "type": "test"} for i in range(100)
    ]
    
    # Process data through system
    insights = []
    for data in test_data:
        # Add to first bank
        if bank1.add_data(data):
            # Get bank positions and resonance fields
            positions = [bank1.position, bank2.position, bank3.position]
            fields = [bank1.resonance_field, bank2.resonance_field, bank3.resonance_field]
            
            # Shift logic path and collect insights
            new_insights = path.shift(positions, fields)
            insights.extend(new_insights)
    
    print(f"Generated {len(insights)} insights")
    for i, insight in enumerate(insights[:3]):
        print(f"\nInsight {i+1}:")
        print(f"Strength: {insight['strength']:.3f}")
        print(f"Position: {insight['pattern']['position']:.2f}")
        print(f"Resonance: {insight['resonance']:.3f}")
