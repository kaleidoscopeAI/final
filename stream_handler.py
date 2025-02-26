import time
import queue
import threading

class StreamHandler:
    def __init__(self, data_processor, batch_size: int = 10, processing_interval: float = 1.0):
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.processing_interval = processing_interval
        self.data_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None

    def add_data(self, data):
        self.data_queue.put(data)

    def start_processing(self):
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._process_data_loop)
            self.processing_thread.start()

    def stop_processing(self):
        if self.is_running:
            self.is_running = False
            if self.processing_thread:
                self.processing_thread.join()

    def _process_data_loop(self):
        while self.is_running:
            batch = []
            while len(batch) < self.batch_size:
                try:
                    data = self.data_queue.get(timeout=self.processing_interval)
                    batch.append(data)
                except queue.Empty:
                    break
            if batch:
                processed_data = self.data_processor.process_and_analyze(np.array(batch))
                print("Processed Data:", processed_data)
            time.sleep(self.processing_interval)
            
    def get_state(self) -> dict:
        return {
            'batch_size': self.batch_size,
            'processing_interval': self.processing_interval,
            'data_queue_size': self.data_queue.qsize(),
            'is_running': self.is_running
        }

