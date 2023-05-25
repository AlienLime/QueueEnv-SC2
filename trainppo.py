
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

# Create an instance of your custom environment
env = QueueEnv()

# Create an algo and configure it with your custom environment
algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .environment(env="QueueEnv")
    .build()
)

# Train the PPO agent
for i in range(5):  # Number of training iterations
    result = algo.train()
    print(result)
    
    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")


