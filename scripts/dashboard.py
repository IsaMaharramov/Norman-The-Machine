import matplotlib.pyplot as plt
from collections import deque

class ReactorDashboard:
    def __init__(self, max_points=200):
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8))
        self.max_points = max_points
        
        self.time = deque(maxlen=max_points)
        self.power = deque(maxlen=max_points)
        self.target = deque(maxlen=max_points)
        self.xenon = deque(maxlen=max_points)
        self.rods = deque(maxlen=max_points)

    def update(self, step, state, target, action):
        self.time.append(step)
        self.power.append(state.power_level)
        self.target.append(target)
        self.xenon.append(state.xenon_conc)
        self.rods.append(action)

        for ax in self.axs: ax.cla()

        # Plot 1: Power vs Target
        self.axs[0].plot(self.time, self.power, label="Actual Power", color='cyan')
        self.axs[0].plot(self.time, self.target, '--', label="Grid Demand", color='orange')
        self.axs[0].set_ylabel("Power %")
        self.axs[0].legend(loc="upper left")

        # Plot 2: Xenon Concentration (The Poison)
        self.axs[1].plot(self.time, self.xenon, color='magenta')
        self.axs[1].axhline(y=5e16, color='red', linestyle='--', label="Poison-out Limit")
        self.axs[1].set_ylabel("Xenon-135")
        
        # Plot 3: Control Rod Position (Agent's Action)
        self.axs[2].step(self.time, self.rods, color='lime')
        self.axs[2].set_ylabel("Rod Pos")
        self.axs[2].set_xlabel("Minutes")

        plt.pause(0.01)