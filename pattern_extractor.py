import numpy as np

class PatternExtractor:
    def __init__(self, data_analyzer: DataAnalyzer):
        self.data_analyzer = data_analyzer
        self.extracted_patterns = []

    def extract_patterns(self, start_time, end_time, pattern_type: str) -> list:
        analysis_results = self.data_analyzer.analyze_historical_data(start_time, end_time)
        if not analysis_results:
            return []
        patterns = []
        if pattern_type == 'trend':
            patterns = self._extract_trends(analysis_results)
        elif pattern_type == 'anomaly':
            patterns = self._extract_anomalies(analysis_results)
        self.extracted_patterns.extend(patterns)
        return patterns

    def _extract_trends(self, analysis_results: dict) -> list:
        trends = []
        for i in range(len(analysis_results['mean'])):
            if analysis_results['mean'][i] > analysis_results['mean'][0] * 1.1:
                trends.append({
                    'type': 'trend',
                    'pattern': 'increasing',
                    'dimension': i,
                    'start_time': analysis_results['start_time'],
                    'end_time': analysis_results['end_time']
                })
            elif analysis_results['mean'][i] < analysis_results['mean'][0] * 0.9:
                trends.append({
                    'type': 'trend',
                    'pattern': 'decreasing',
                    'dimension': i,
                    'start_time': analysis_results['start_time'],
                    'end_time': analysis_results['end_time']
                })
        return trends

    def _extract_anomalies(self, analysis_results: dict) -> list:
        anomalies = []
        data = self.data_analyzer.data_storage.retrieve_data(analysis_results['start_time'], analysis_results['end_time'])
        data_array = np.array(data)
        for i in range(data_array.shape[1]):
            for j, value in enumerate(data_array[:, i]):
                z_score = (value - analysis_results['mean'][i]) / analysis_results['std'][i] if analysis_results['std'][i] > 0 else 0
                if abs(z_score) > 2.5:
                    anomalies.append({
                        'type': 'anomaly',
                        'dimension': i,
                        'value': value,
                        'z_score': z_score,
                        'timestamp': data[j]['timestamp'],
                        'data_index': j
                    })
        return anomalies

    def get_state(self) -> dict:
        return {
            'extracted_patterns_count': len(self.extracted_patterns),
            'last_extraction': {
                'start_time': self.extracted_patterns[-1]['start_time'] if self.extracted_patterns else None,
                'end_time': self.extracted_patterns[-1]['end_time'] if self.extracted_patterns else None,
                'pattern_type': self.extracted_patterns[-1]['type'] if self.extracted_patterns else None
            }
        }

