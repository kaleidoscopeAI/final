import numpy as np

class DataProcessor:
    def __init__(self, kaleidoscope_engine, perspective_engine):
        self.kaleidoscope_engine = kaleidoscope_engine
        self.perspective_engine = perspective_engine

    def process_data(self, data: np.ndarray) -> dict:
        kaleidoscope_output = self.kaleidoscope_engine.process_data(data)
        perspective_output = self.perspective_engine.process_data(data)
        combined_output = {
            'kaleidoscope': kaleidoscope_output,
            'perspective': perspective_output
        }
        return combined_output

    def process_and_analyze(self, data: np.ndarray) -> dict:
        kaleidoscope_output = self.kaleidoscope_engine.process_data(data)
        perspective_output = self.perspective_engine.process_data(data)
        combined_output = {
            'kaleidoscope': kaleidoscope_output,
            'perspective': perspective_output
        }
        return combined_output
    
    def get_state(self) -> dict:
        return {
            'kaleidoscope_engine_state': self.kaleidoscope_engine.get_state(),
            'perspective_engine_state': self.perspective_engine.get_state()
        }

