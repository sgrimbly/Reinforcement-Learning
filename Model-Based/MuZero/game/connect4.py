# https://github.com/plkmo/AlphaZero_Connect4/blob/master/src/connect_board.py
from typing import Dict, List, Optional
from game.game import Action, AbstractGame
import numpy as np
import enum

Player = enum.Enum("Player", "black white")
Winner = enum.Enum("Winner", "black white draw")


class ActionHistory(object):
    """Simple history container used inside the search.
    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)

        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [i for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        if len(self.history) % 2 == 0:
            return Player.white
        else:
            return Player.black

class Environment(object):
    """The environment MuZero is interacting with."""
    def __init__(self):
        self.board = None
        self.turn = 0
        self.done = False
        self.winner = None  # type: Winner
        self.resigned = False

    def reset(self):
        self.board = []
        for i in range(6):
            self.board.append([])
            for j in range(7): # pylint: disable=unused-variable
                self.board[i].append(' ')
        self.turn = 0
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def update(self, board):
        self.board = np.copy(board)
        self.turn = self.turn_n()
        self.done = False
        self.winner = None
        self.resigned = False
        return self

    def turn_n(self):
        turn = 0
        for i in range(6):
            for j in range(7):
                if self.board[i][j] != ' ':
                    turn += 1

        return turn

    def player_turn(self):
        if self.turn % 2 == 0:
            return Player.white
        else:
            return Player.black

    def step(self, action):
        for i in range(6):
            if self.board[i][action] == ' ':
                self.board[i][action] = ('X' if self.player_turn() == Player.white else 'O')
                break

        self.turn += 1

        self.check_for_fours()

        if self.turn >= 42:
            self.done = True
            if self.winner is None:
                self.winner = Winner.draw

        r = 0
        if self.done:
            if self.turn % 2 == 0:
                if Winner.white:
                    r = 1
                elif Winner.black:
                    r = -1
            else:
                if Winner.black:
                    r = 1
                elif Winner.white:
                    r = -1

        return r

    def legal_moves(self):
        legal = [0, 0, 0, 0, 0, 0, 0]
        for j in range(7):
            for i in range(6):
                if self.board[i][j] == ' ':
                    legal[j] = 1
                    break
        return legal

    def legal_actions(self):
        legal = []
        for j in range(7):
            for i in range(6):
                if self.board[i][j] == ' ':
                    legal.append(j)
                    break
        return legal

    def check_for_fours(self):
        for i in range(6):
            for j in range(7):
                if self.board[i][j] != ' ':
                    # check if a vertical four-in-a-row starts at (i, j)
                    if self.vertical_check(i, j):
                        self.done = True
                        return

                    # check if a horizontal four-in-a-row starts at (i, j)
                    if self.horizontal_check(i, j):
                        self.done = True
                        return

                    # check if a diagonal (either way) four-in-a-row starts at (i, j)
                    diag_fours = self.diagonal_check(i, j)
                    if diag_fours:
                        self.done = True
                        return

    def vertical_check(self, row, col):
        # print("checking vert")
        four_in_a_row = False
        consecutive_count = 0

        for i in range(row, 6):
            if self.board[i][col].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break

        if consecutive_count >= 4:
            four_in_a_row = True
            if 'x' == self.board[row][col].lower():
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        return four_in_a_row

    def horizontal_check(self, row, col):
        four_in_a_row = False
        consecutive_count = 0

        for j in range(col, 7):
            if self.board[row][j].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break

        if consecutive_count >= 4:
            four_in_a_row = True
            if 'x' == self.board[row][col].lower():
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        return four_in_a_row

    def diagonal_check(self, row, col):
        four_in_a_row = False
        count = 0

        consecutive_count = 0
        j = col
        for i in range(row, 6):
            if j > 6:
                break
            elif self.board[i][j].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break
            j += 1

        if consecutive_count >= 4:
            count += 1
            if 'x' == self.board[row][col].lower():
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        consecutive_count = 0
        j = col
        for i in range(row, -1, -1):
            if j > 6:
                break
            elif self.board[i][j].lower() == self.board[row][col].lower():
                consecutive_count += 1
            else:
                break
            j += 1

        if consecutive_count >= 4:
            count += 1
            if 'x' == self.board[row][col].lower():
                self.winner = Winner.white
            else:
                self.winner = Winner.black

        if count > 0:
            four_in_a_row = True

        return four_in_a_row

    def black_and_white_plane(self):
        board_white = np.copy(self.board)
        board_black = np.copy(self.board)
        for i in range(6):
            for j in range(7):
                if self.board[i][j] == ' ':
                    board_white[i][j] = 0
                    board_black[i][j] = 0
                elif self.board[i][j] == 'X':
                    board_white[i][j] = 1
                    board_black[i][j] = 0
                else:
                    board_white[i][j] = 0
                    board_black[i][j] = 1

        return np.array(board_white), np.array(board_black)

    def render(self):
      print("\nRound: " + str(self.turn))

      for i in range(5, -1, -1):
          print("\t", end="")
          for j in range(7):
              print("| " + str(self.board[i][j]), end=" ")
          print("|")
      print("\t  _   _   _   _   _   _   _ ")
      print("\t  1   2   3   4   5   6   7 ")

      if self.done:
          print("Game Over!")
          if self.winner == Winner.white:
              print("X is the winner")
          elif self.winner == Winner.black:
              print("O is the winner")
          else:
              print("Game was a draw")

    @property
    def observation(self):
        return ''.join(''.join(x for x in y) for y in self.board)

class Connect4(AbstractGame):
    """A single episode of interaction with the environment."""
    def __init__(self, action_space_size: int, discount: float):
        self.environment = Environment().reset()  # Game specific environment.
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        
        self.action_space_size = action_space_size
        self.discount = discount

    def terminal(self) -> bool:
        # Game specific termination rules.
        return self.environment.done

    def legal_actions(self) -> List[Action]:
        # Game specific calculation of legal actions.
        return self.environment.legal_actions()

    def apply(self, action: Action):
        reward = self.environment.step(action)
        reward = reward if self.environment.turn % 2 != 0 and reward == 1 else -reward
        self.rewards.append(reward)
        self.history.append(action)

    def make_image(self, state_index: int):
        # Game specific feature planes.    
        o = Environment().reset()

        for current_index in range(0, state_index):
            o.step(self.history[current_index])

        black_ary, white_ary = o.black_and_white_plane()
        state = [black_ary, white_ary] if o.player_turn() == Player.black else [white_ary, black_ary]
        return np.array(state)

    def make_target(self, state_index: int, num_unroll_steps: int, td_steps: int,
                    to_play: Player):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
        if bootstrap_index < len(self.root_values):
            value = self.root_values[bootstrap_index] * self.discount**td_steps
        else:
            value = 0

        for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
            value += reward * self.discount**i  # pytype: disable=unsupported-operands

        if current_index < len(self.root_values):
            targets.append((value, self.rewards[current_index],
                            self.child_visits[current_index]))
        else:
            # States past the end of games are treated as absorbing states.
            targets.append((0, 0, []))
        return targets

    def to_play(self) -> Player:
        return self.environment.player_turn

    def action_space_size(self) -> int:
        """Return the size of the action space."""
        return self.action_space_size

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

    def step(self, action) -> int:
        """Execute one step of the game conditioned by the given action."""
        return Environment.step(self, action)






    '''
    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in action_space
        ])
        self.root_values.append(root.value())
    '''

