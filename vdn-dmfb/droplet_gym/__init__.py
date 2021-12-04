from gym.envs.registration import register

register(
    id='droplet-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='droplet_gym.envs:Dropletenv',              # Expalined in envs/__init__.py
)