# QueueEnv-SC2
A Gymnasium environment for training SC2 agents with reinforcement learning

To utilize this environment you will need to install the following packages for Python 3.10.10:

    pip install ray

    pip install "ray[rllib]" torch

    pip install gymnasium

    pip install wandB
        Remember to login with wandb login and your API key

    pip install --upgrade burnysc2


Then you will need to:

    Correct 2 signals in API:
        Outcomment line 104 and 119 in sc2process

    Get the maps and put them in the right folder. On windows this would often be:
        C:\Program Files (x86)\StarCraft II\Maps

Now simply run QueueEnv.py to train the agent.
    Remember to choose the correct project and map



