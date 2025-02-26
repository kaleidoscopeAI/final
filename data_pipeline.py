class DataPipeline:
    def __init__(self, stream_handler, data_storage, historical_analyzer, pattern_extractor):
        self.stream_handler = stream_handler
        self.data_storage = data_storage
        self.historical_analyzer = historical_analyzer
        self.pattern_extractor = pattern_extractor

    def run_pipeline(self):
        self.stream_handler.start_processing()
        import time
        while True:
            time.sleep(3600)
            self.analyze_historical_data()

    def analyze_historical_data(self):
        end_time = time.time()
        start_time = end_time - 86400
        analysis_results = self.historical_analyzer.analyze_historical_data(start_time, end_time)
        print("Historical Analysis Results:", analysis_results)
        extracted_patterns = self.pattern_extractor.extract_patterns(start_time, end_time, 'trend')
        print("Extracted Trend Patterns:", extracted_patterns)
        anomalies = self.pattern_extractor.extract_patterns(start_time, end_time, 'anomaly')
        print("Extracted Anomalies:", anomalies)
        
    def get_state(self) -> dict:
        return {
            'stream_handler_state': self.stream_handler.get_state(),
            'data_storage_state': self.data_storage.get_state(),
            'historical_analyzer_state': self.historical_analyzer.get_state(),
            'pattern_extractor_state': self.pattern_extractor.get_state()
        }

