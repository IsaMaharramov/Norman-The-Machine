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
        
        self.line_power, = self.axs[0].plot([], [], label="Actual Power", color='cyan')
        self.line_target, = self.axs[0].plot([], [], '--', label="Grid Demand", color='orange')
        self.axs[0].set_ylabel("Power %")
        self.axs[0].legend(loc="upper left")

        self.line_xenon, = self.axs[1].plot([], [], color='magenta')
        self.axs[1].axhline(y=5e16, color='red', linestyle='--', label="Poison-out Limit")
        self.axs[1].set_ylabel("Xenon-135")
        
        self.line_rods, = self.axs[2].plot([], [], color='lime')
        self.axs[2].set_ylabel("Rod Pos")
        self.axs[2].set_xlabel("Minutes")

    def update(self, step, state, target, action):
        self.time.append(step)
        self.power.append(state.power_level)
        self.target.append(target)
        self.xenon.append(state.xenon_conc)
        self.rods.append(action)

        t_list = list(self.time)

        self.line_power.set_data(t_list, list(self.power))
        self.line_target.set_data(t_list, list(self.target))
        self.line_xenon.set_data(t_list, list(self.xenon))
        self.line_rods.set_data(t_list, list(self.rods))

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
            if t_list:
                ax.set_xlim(t_list[0], t_list[-1] + 1)
        
        self.axs[2].set_ylim(-0.05, 1.05)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)