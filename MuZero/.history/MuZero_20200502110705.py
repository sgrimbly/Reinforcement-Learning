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

# Each instantiation of self-play is an independent job. The latest network is taken and plays a game. The played game is then saved to the shared storage.
# What this means is, MuZero plays num_actors number of games against itself using the latest network. These are saved for learning.
def run_selfplay(config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
    while True:
        network = storage.latest_network
        game = play_game(config, network)
        replay_buffer = save_game(game)


class SharedStorage(object):
    def __init__(self):
        self._networks = {}
    
    def latest_network(self) -> Network: # -> specifies return type
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network()

    def save_network(self, step: int, network: Network):
        self._networks[step] = network

class ReplayBuffer(object):
    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size # The maximum number of games stored in the buffer
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

# The exact models aren’t provided in the pseudocode, but detailed descriptions are given in the accompanying paper.
class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]

class Network(object):
    def initial_inference(self,image) -> NetworkOutput:
        # The representation and prediction function
        return NetworkOutput(0,0,{},[])

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # The dynamics and prediction function
        return NetworkOutput(0,0,{},[])

    def get_weights(self):
        # Returns the weights of this network.
        return []

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0

# PART 2: https://medium.com/applied-data-science/how-to-build-your-own-deepmind-muzero-in-python-part-2-3-f99dad7a7ad

def play_game(config: MuZeroConfig, network: Network) -> Game:
    game = config.new_game()

    while not game.terminal() and len(game.history) < config.max_moves:
        # Use the representation function at the root of the search tree to obtain the hidden (latent) state given the current observation
        root = Node(0)
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(), network.initial_inference(current_observation))
        add_exploration_noise(config, root)

        # Run MCTS using action sequences and the learned model
        run_mcts(config, root, game.action_history(), network)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
        
    return game

class Node(object):
    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum/self.visit_count

def expand_node(node: Node, Player: player, actions: List[Action], network_output: NetworkOutput):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions} 
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)
        