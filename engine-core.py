class MemoryBank:
    def __init__(self):
        self.data = []
        self.position = 0
        self.connected_paths = []
        self.threshold = 100

    def add_data(self, data):
        self.data.append(data)
        if len(self.data) >= self.threshold:
            self._rotate()
            return True
        return False

    def _rotate(self):
        # Simulate gear rotation by shifting position
        self.position = (self.position + 45) % 360
        # Update connected paths
        for path in self.connected_paths:
            path.shift(self.position)
        # Clear processed data
        self.data = []

class LogicPath:
    def __init__(self):
        self.position = 0
        self.insights = []

    def shift(self, bank_position):
        # Position changes based on connected bank rotation
        self.position = (self.position + bank_position) % 360
        self._generate_insights()

    def _generate_insights(self):
        # Generate insights based on current position
        # Different positions yield different insight combinations
        sector = self.position // 45  # 8 sectors of 45 degrees each
        # Map sector to insight generation logic
        self.insights.append({
            'sector': sector,
            'pattern': f"Pattern_{sector}",
            'weight': sector / 8
        })

class Engine:
    def __init__(self, is_ethical=True):
        self.banks = [MemoryBank() for _ in range(5)]
        self.paths = [LogicPath() for _ in range(4)]
        self.is_ethical = is_ethical
        
        # Connect banks to paths
        for bank in self.banks:
            bank.connected_paths = self.paths

    def process_data(self, data):
        processed_data = self._filter_data(data) if self.is_ethical else data
        
        # Add data to banks and collect insights
        insights = []
        for bank in self.banks:
            if bank.add_data(processed_data):
                # Collect insights when bank rotates
                for path in self.paths:
                    insights.extend(path.insights)
                    path.insights = []  # Clear processed insights
        
        return insights

    def _filter_data(self, data):
        if not self.is_ethical:
            return data
        # Apply ethical constraints for Kaleidoscope engine
        if isinstance(data, dict):
            filtered = data.copy()
            if 'risk' in filtered:
                filtered['risk'] = min(filtered['risk'], 0.8)
            return filtered
        return data

    def crystallize(self, nodes, insights):
        """Create perfect representative node from processed data."""
        # Combine all insights and node data
        all_data = []
        for node in nodes:
            all_data.extend(node.memory)
        all_data.extend(insights)

        # Process through banks one final time
        final_insights = []
        for data in all_data:
            final_insights.extend(self.process_data(data))

        # Create perfect representative node
        perfect_node = Node()
        perfect_node.memory = final_insights
        return perfect_node

class Node:
    def __init__(self):
        self.memory = []
        self.threshold = 100

    def add_data(self, data):
        self.memory.append(data)
        return len(self.memory) >= self.threshold

