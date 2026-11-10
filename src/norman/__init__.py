from gymnasium.envs.registration import register

__version__ = "1.0.0"

register(id="Norman/LoadFollow-v1", entry_point="norman.envs.load_follow:LoadFollowEnv")
