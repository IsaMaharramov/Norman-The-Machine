import gymnasium as gym
from gymnasium import spaces
import numpy as np
import norman_core
from envs.demand import DemandGenerator

class NormanReactorEnv(gym.Env):
    def __init__(self, dt=10.0): 
        super(NormanReactorEnv, self).__init__()
        self.engine = norman_core.ReactorEngine(dt)
        self.demand_gen = DemandGenerator()
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        self.target_power = 1.0 
        self.steps_taken = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset()
        self.steps_taken = 0
        self.target_power = self.demand_gen.get_target(0)
        
        state = self.engine.get_state()
        obs = self._get_obs(state)
        return obs, {}

    def _get_obs(self, state):
        flux = np.clip(state.neutron_flux, 1e-1, 1e20)
        iodine = np.clip(state.iodine_conc, 1e-1, 1e20)
        xenon = np.clip(state.xenon_conc, 1e-1, 1e20)
        
        return np.array([
            np.log10(flux),             
            np.log10(iodine),           
            np.log10(xenon),            
            np.clip(state.power_level, 0.0, 5.0), 
            self.target_power
        ], dtype=np.float32)

    def step(self, action):
        self.engine.step(float(action[0]))
        
        state = self.engine.get_state() 
        self.steps_taken += 1
        
        self.target_power = self.demand_gen.get_target(self.steps_taken)
        reward = self._calculate_reward(state)
        
        poisoned = bool(state.xenon_conc > 5e16) 
        math_fail = np.isnan(state.power_level) or np.isinf(state.power_level)
        
        terminated = poisoned or math_fail
        truncated = self.steps_taken >= 1440
        
        return self._get_obs(state), reward, terminated, truncated, {}

    def _calculate_reward(self, state):
        if np.isnan(state.power_level) or np.isinf(state.power_level):
            return -500.0

        error = abs(state.power_level - self.target_power)
        r_accuracy = -30.0 * (error ** 2)
        
        poison_limit = 5e16
        xenon_ratio = np.clip(state.xenon_conc / poison_limit, 0.0, 1.0)
        r_safety = -5.0 * (np.exp(6.0 * xenon_ratio) - 1.0)
        
        return r_accuracy + r_safety