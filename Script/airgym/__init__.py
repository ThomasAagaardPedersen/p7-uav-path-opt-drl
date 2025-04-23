from gym.envs.registration import register

register(
    id="airsim-drone-sample-v0", entry_point="airgym.envs:AirSimDroneEnv",
)

register(
    id="airsim-car-sample-v0", entry_point="airgym.envs:AirSimCarEnv",
)

#register(id="gym-AirSimWindEnv-0.0.1", entry_point="airgym.envs:AirSimWindEnv",)
