# Copyright 2019 Ioannis Arailopoulos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""API to transform communications between a Poker Gym environment
and the Agents.

This API is based on the gym holdem package 
"""

import collections
import math

import numpy as np
import gym

from pokermon.envs import holdem

class PokerGymSimulator():
    """Environment simulation API for the agents"""
    def __init__(self, env):
        self.env = env
        self.env.add_player(0, stack=2000) # add a player to seat 0 with 2000 "chips"
        self.env.add_player(1, stack=2000) # add another player to seat 1 with 2000 "chips"
        self.env.reset()
        current_player = self.env._current_player
        state = self.env._output_state(current_player)
        self.stack = int(state.get('stack'))
        self.action_history = "p"
        self.flop = False
        self.turn = False
        self.river = False

    def get_action_space(self):
        """Return the available actions for the current player"""
        current_player = self.env._current_player
        state = self.env._output_state(current_player)
        minraise =  state.get('minraise', 0)
        maxraise = state.get('stack')
        bblind = state.get('bigblind')
        tocall = min(state.get('tocall', 0), state.get('stack'))
        raise_buckets = self.get_raise_amount_buckets(minraise, maxraise, bblind)
        raise_amount = ['bet' + str(i) for i in raise_buckets]
        if tocall > 0:
            action_space = ['fold', 'call'] + raise_amount
        else:
            action_space = ['call'] + raise_amount
        return tuple(action_space)

    def get_raise_amount_buckets(self, minraise, maxraise, bblind):
        """Bucketize the bet amounts"""
        raise NotImplementedError

    def get_history(self, action=None):
        """Return the game history for the CURRENT_PLAYER"""
        player = self.env._current_player
        state = self.env._output_state(player)
        player_cards = self._get_player_cards(state)
        community_cards = self._get_community_cards(state)
        if action is not None:
            self._update_action_history(action, player)
        return (tuple(player_cards), tuple(community_cards), 
               self.action_history)

    def _get_player_cards(self, state):
        """Return the cards of the current player"""
        return state.get('pocket_cards', -1)

    def _get_community_cards(self, state):
        """Return the community cards"""
        return state.get('community')

    def _update_action_history(self, action, player):
        """Update the ACTION_HISTORY for the PLAYER"""
        if not self.flop:
            if self.env._round == 1:
                self.flop = True
                self.action_history += "-f"
        elif not self.turn:
            if self.env._round == 2:
                self.turn = True
                self.action_history += "-t"
        elif not self.river:
            if self.env._round == 3:
                self.river = True
                self.action_history += "-r"
        self.action_history += "-" + action + "-" + str(player.player_id)

    def get_current_player(self):
        """Return the current player id"""
        return self.env._current_player.player_id

    def step(self, action):
        """Return the state, reward, done, info after the current player
        taking ACTION"""
        env_action = self._convert_raw_to_env_action(action)
        return self.env.step(env_action)

    def reset(self):
        """Reset the environment."""
        self.action_history = 'p'
        self.flop = False
        self.turn = False
        self.river = False
        self.env.remove_player(0)
        self.env.remove_player(1)
        self.env.add_player(0, stack=100) # add a player to seat 0 with 2000 "chips"
        self.env.add_player(1, stack=100) # add another player to seat 1 with 2000 "chips"
        self.env.reset()

    def _convert_raw_to_env_action(self, action):
        """Converts a raw action from the Agent to an environment action"""
        current_player = self.env._current_player
        current_player_id = self.get_current_player()
        actions = [[0, 0]] * self.env.n_seats
        if action == 'fold':
            actions[current_player_id] = [3, 0]
        elif action == 'call':
            state = self.env._output_state(current_player)
            tocall = min(state.get('tocall', 0), state.get('stack'))
            actions[current_player_id] = [1, tocall] if tocall > 0 else [0, 0]
        else:
            raise_amount = action[3:]
            actions[current_player_id] = [2, raise_amount]
        return actions
