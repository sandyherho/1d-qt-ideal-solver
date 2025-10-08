import time
from contextlib import contextmanager

class Timer:
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start(self, name):
        self.start_times[name] = time.time()
    
    def stop(self, name):
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.times[name] = elapsed
            del self.start_times[name]
            return elapsed
        return 0
    
    @contextmanager
    def time_section(self, name):
        self.start(name)
        yield
        self.stop(name)
    
    def get_times(self):
        return self.times.copy()
