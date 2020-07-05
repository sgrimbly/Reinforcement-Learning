from config import MuZeroConfig, make_cartpole_config
from networks.shared_storage import SharedStorage
from self_play.self_play import run_selfplay, run_eval
from training.replay_buffer import ReplayBuffer
from training.training import train_network

import matplotlib.pyplot as plt

def muzero(config: MuZeroConfig):
    """
    MuZero training is split into two independent parts: Network training and
    self-play data generation.
    These two parts only communicate by transferring the latest networks checkpoint
    from the training to the self-play, and the finished games from the self-play
    to the training.
    In contrast to the original MuZero algorithm this version doesn't works with
    multiple threads, therefore the training and self-play is done alternately.
    """
    storage = SharedStorage(config.new_network(), config.uniform_network(), config.new_optimizer())
    replay_buffer = ReplayBuffer(config)

    scores = []
    for loop in range(config.nb_training_loop):
        print("Training loop", loop)
        score_train = run_selfplay(config, storage, replay_buffer, config.nb_episodes)
        train_network(config, storage, replay_buffer, config.nb_epochs)

        scores.append([loop,score_train])

        print("Train score:", score_train)
        print("Eval score:", run_eval(config, storage, 50))
        print(f"MuZero played {config.nb_episodes * (loop + 1)}" f"episodes and trained for {config.nb_epochs * (loop + 1)} epochs.\n")

    return scores, storage.latest_network()


if __name__ == '__main__':
    config = make_cartpole_config()
    scores, latest_network = muzero(config)

    plt.plot(scores[0], scores[1], linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Training Score")
    plt.title("Training score of MuZero agent learning CartPole over time")
    plt.show()
