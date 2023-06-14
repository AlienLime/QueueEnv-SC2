import wandb





print(min(140, 999))

"""
# Initialize WandB
wandb.init(project="test")

rewards =[2,5,3,2,5]
data = []

for i in range(len(rewards)):
    data.append([i, rewards[i]])

print(data)

table = wandb.Table(data = data, columns = ["Episodes", "Rewards"])

wandb.log({"episode_rewards" : wandb.plot.line(table, "Episodes", "Rewards", title="Custom Episode Rewards Line Plot")})

wandb.finish()
"""

"""
episode_reward = result["hist_stats"]["episode_reward"]

data = []

for ep in range(len(episode_reward)):
    data.append([ep, episode_reward[ep]])
print(data)    

table = wandb.Table(data = data, columns = ["Episodes", "Rewards"])

wandb.log({"episode_rewards" : wandb.plot.line(table, "Episodes", "Rewards", title="Custom Episode Rewards Line Plot")})
"""