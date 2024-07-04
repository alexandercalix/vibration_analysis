import random

class SPIReader:
    def __init__(self, fault_probability=0.1):
        self.fault_probability = fault_probability

    def read_vibration_data(self):
        if random.random() < self.fault_probability:
            # Generate fault data
            return [random.uniform(-0.05, 0.05) for _ in range(3)]
        else:
            # Generate normal data
            return [random.uniform(-0.0005, 0.0005) for _ in range(3)]

    def close(self):
        pass
