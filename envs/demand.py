import numpy as np

class DemandGenerator:
    def __init__(self):
        self.base_load = 0.6
    
    def get_target(self, step):
        # 1 step = 1 minute
        time = (step % 1440) / 1440.0
        
        # Diurnal cycle (Sine wave) + evening peak
        cycle = 0.2 * np.sin(2 * np.pi * time - np.pi/2)
        peak = 0.1 if 0.7 < time < 0.9 else 0.0
        
        noise = np.random.normal(0, 0.015)
        
        return np.clip(self.base_load + cycle + peak + noise, 0.1, 1.0)