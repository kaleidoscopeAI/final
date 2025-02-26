import numpy as np

class DataAnalyzer:
    def __init__(self, data_storage):
        self.data_storage = data_storage

    def analyze_historical_data(self, start_time, end_time) -> dict:
        data = self.data_storage.retrieve_data(start_time, end_time)
        if not data:
            return {}
        data_array = np.array(data)
        analysis = {
            'start_time': start_time,
            'end_time': end_time,
            'data_points': len(data_array),
            'mean': np.mean(data_array, axis=0).tolist(),
            'std': np.std(data_array, axis=0).tolist(),
            'min': np.min(data_array, axis=0).tolist(),
            'max': np.max(data_array, axis=0).tolist(),
            'correlations': self._calculate_correlations(data_array)
        }
        return analysis

    def _calculate_correlations(self, data_array: np.ndarray) -> dict:
        num_dimensions = data_array.shape[1]
        correlations = {}
        for i in range(num_dimensions):
            for j in range(i+1, num_dimensions):
                correlation = np.corrcoef(data_array[:, i], data_array[:, j])[0,1]
                correlations[f'dim_{i}_vs_dim_{j}'] = correlation
        return correlations

    def get_state(self) -> dict:
        return {
            'data_storage_size': len(self.data_storage.data),
            'last_analysis': self.analyze_historical_data(self.data_storage.data[0]['timestamp'], self.data_storage.data[-1]['timestamp']) if self.data_storage.data else None
        }

