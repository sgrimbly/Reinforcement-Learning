# A simple implementation of the MuZero algorithm presented by DeepMind in Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
# Original Paper: https://arxiv.org/abs/1911.08265 
# Pseudocode: https://arxiv.org/src/1911.08265v1/anc/pseudocode.py
# Adapted from: https://medium.com/applied-data-science/how-to-build-your-own-muzero-in-python-f77d5718061a
# Author: St John Grimbly

def muzero(config: MuZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    for _ in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer)
        
    train_network(config, storage, replay_buffer)

    return storage.latest_network()
    
