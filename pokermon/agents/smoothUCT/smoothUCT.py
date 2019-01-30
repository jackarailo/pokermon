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

"""Monte Carlo Tree Search implementation for PODMP.

Borrowing ideas from SmoothUCT Search in Computer Poker
from Johannes Heinrich and David Silver, Monte-Carlo
Planning in Large POMDPs from David Silver and Joel Veness,
and https://github.com/tensorflow/minigo implementation
"""

import collections
import math

import numpy as np
import pickle
import json
from tqdm import tqdm

def get_rollout_action(simulator, mode='random'):
    """Return the action based on the rollout policy"""
    if mode=='random':
        return rollout_policy_random(simulator)
    elif mode=='rnn-approximation-function':
        return rollout_policy_rnn(simulator)

def rollout_policy_random(simulator):
    """Return a random action during the rollout policy

    Assumptions: The simulator returns actions as a tuple 
    (fold, check, bet1, bet2, ...).

    The random policy chooses equally between the actions fold, check, and bet.
    If the choosen action is bet the random policy chooses equally among the 
    bet amounts.
    """
    action_space = simulator.get_action_space()
    action_idx = list(range(len(action_space)))
    nactions = len(action_idx)
    std_actions = 2 if action_space[0] == 'fold' else 1
    p = [(3*(nactions-std_actions))**-1 if i > 1 else 1./(std_actions+1) for i in range(nactions)]
    if np.sum(p) != 1.0:
        p = p/np.sum(p)
    best_action_idx = np.random.choice(action_idx, p=p)
    return action_space[best_action_idx]

def rollout_policy_rnn(simulator):
    """TODO: Return a rollout policy based on an RNN (LSTM) network"""
    raise NotImplementedError

class MCTSHeadNode():
    """Monte Carlo Tree Search Head (Chance) Node"""
    def __init__(self, simulator, select_mode='UCT', 
                rollout_policy_mode='random', C=1.75):
        self.select_mode = select_mode
        self.rollout_policy_mode = rollout_policy_mode
        self.C = C
        self.simulator = simulator
        self.simulator.reset()
        self.children = {}

    def simulate(self):
        history = self.simulator.get_history()
        if history in self.children.keys():
            self.children[history].simulate()
        else:
            self.children[history] = MCTSNode(
                                      history, 
                                      simulator=self.simulator,
                                      parent=None, 
                                      parent_action=None,
                                      select_mode=self.select_mode,
                                      rollout_policy_mode=self.rollout_policy_mode,
                                      C=self.C)
            self.children[history].simulate()

    def get_policy(self):
        self.policy = {}
        for history in self.children.keys():
            current_node = self.children[history]
            self.policy[history] = current_node.get_policy()
            self.update_policy(current_node)
        return self.policy

    def update_policy(self, current_node):
        for child in current_node.children.keys():
            self.update_policy(current_node.children[child])
        self.policy[current_node.history] = current_node.get_policy()

class MCTSNode():
    """Monte Carlo Tree Search Node to keep track of the statistics"""
    def __init__(self, history, simulator, parent=None, 
                parent_action=None, select_mode='UCT', 
                rollout_policy_mode='random', C=1.75):
        """Initialize node"""
        self.simulator = simulator
        self.player = self.simulator.get_current_player()
        self.history = history
        self.parent = parent
        self.parent_action = parent_action
        self.rollout_policy_mode = rollout_policy_mode
        self.select_mode = select_mode
        self.C = C
        self.children = {}
        self.N = {}
        self.N[None] = 0
        self.Q = {}
        self.Q_cum = {}
        for action in self.simulator.get_action_space():
            self.N[action] = 0
            self.Q[action] = 0.0
            self.Q_cum[action] = 0.0

    def hasChildren(self, action=None, history=None):
        """Return False if the node is a leaf else True"""
        if action is None and history is None:
            return bool(self.children)
        else:
            return bool(history in self.children.keys())

    def hasParent(self):
        """Return True if the node is not the head else False"""
        return True if self.parent is not None else False

    def simulate(self):
        """Select the MTCSNode based on the UCB criterion"""
        action = self.select()
        new_history = self.simulator.get_history(action)
        _, reward, done, _ = self.simulator.step(action)
        if done:
            self.backup(reward, action)
        elif self.hasChildren(new_history):
            self.children[new_history].simulate()
        else:
            current_node = self.expand(action, new_history) 
            new_action = get_rollout_action(self.simulator, 
                                            self.rollout_policy_mode)
            reward = current_node.rollout(new_action)
            current_node.backup(reward, new_action)
        self.simulator.reset()

    def select(self):
        """Return the best action"""
        if self.N[None] == 0:
            action_space = list(self.Q.keys())
            action_idx = list(range(len(action_space)))
            nactions = len(action_idx)
            std_actions = 2 if 'fold' in action_space else 1
            p = [(3*(nactions-std_actions))**-1 if i > 1 else 1./(std_actions+1) for i in range(nactions)]
            if np.sum(p) != 1.0:
                p = p/np.sum(p)
            best_action_idx = np.random.choice(action_idx, p=p)
            return action_space[best_action_idx]
        elif self.select_mode == 'UCT':
            return self.best_action_uct()
        elif self.select_mode == 'smoothUCT':
            return self.best_action_smooth_uct()

    def expand(self, action, new_history):
        """Expand node from ACTION and return new MCTSNODE"""
        self.children[new_history] = MCTSNode(
                                  new_history, 
                                  simulator=self.simulator,
                                  parent=self, 
                                  parent_action=action,
                                  select_mode=self.select_mode,
                                  rollout_policy_mode=self.rollout_policy_mode,
                                  C=self.C)
        return self.children[new_history]

    def rollout(self, action):
        """Rollout from node based on rollout policy"""
        _, reward, done, _ = self.simulator.step(action)
        while not done:
            action = get_rollout_action(self.simulator,
                                        self.rollout_policy_mode)
            _, reward, done, _ = self.simulator.step(action)
        return reward

    def backup(self, reward, action):
        """Backup the reward in the tree following a rollout"""
        self.N[None] += 1
        self.N[action] += 1
        self.Q_cum[action] = reward[self.player]
        self.Q[action] = self.Q_cum[action]/self.N[action]
        if self.hasParent():
            self.parent.backup(reward, self.parent_action)

    def best_action_uct(self):
        """Return the best action based on the UCT method"""
        max_uct = -np.Inf
        logsumN = np.log(self.N[None])
        for action in self.Q.keys():
            q = self.Q[action]
            if self.N[action] > 0:
                u = self.C*np.sqrt(logsumN/self.N[action]) 
            else:
                u = 0
            if q+u > max_uct:
                best_actions = [action]
            elif q+u == max_uct:
                best_actions.append(action)
        return np.random.choice(best_actions)

    def best_action_smooth_uct(self):
        """Return the best action based on the smoothUCT method"""
        # Using constants with best results accoridng to Heinrich
        gamma = 0.1 
        eta0 = 0.9
        d = 0.001 
        sqsumN = np.sqrt(self.N[None])
        eta = np.max(gamma, eta0/(1+d*sqsumN))
        if np.random.uniform(low=0.0, high=1.0) < eta:
            return best_action_uct()
        else:
            actions = self.Q.keys()
            p = np.zeros(shape=len(actions))
            choice_idx = np.arange(shape=len(actions))
            idx_to_action = {}
            i = 0
            for action in actions:
                idx_to_action[i] = action
                p[i] = self.N[action] / self.N[None]
            return idx_to_action[np.random.choice(choice_idx, p)]

    def get_policy(self):
        """Return the best policy learned so far based on action counts"""
        max_a = -np.Inf
        for action in self.Q.keys():
            temp = self.N[action]
            if temp > max_a:
                max_a = temp
                best_a = action
        return best_a if max_a > 0 else np.random.choice(self.Q.keys())

class smoothUCT():
    """Monte Carlo Tree Search Implementation"""
    def __init__(self, simulator, select_mode='UCT', 
                 save_policy_step=100, rollout_policy_mode='random', 
                 C=1.75):
        """Initialize Monte Carlo Tree Search"""
        self.select_mode = select_mode
        self.rollout_policy_mode = rollout_policy_mode
        self.simulator = simulator
        self.head_node = MCTSHeadNode(self.simulator, select_mode, 
                                      rollout_policy_mode, C)
        self.save_policy_step = save_policy_step

    def search(self, iters=1000):
        """Update the tree based on the new history O"""
        # Monte Carlo Tree Search
        for i in tqdm(range(iters)):
            self.head_node.simulate()
            if i % self.save_policy_step == 0:
                self.save_policy(i)

    def save_policy(self, i):
        """Save policy to JSON file"""
        policy = self.head_node.get_policy()
        filename = "smoothUCT_" + str(i) + ".txt"
        with open(filename, 'w') as f:
            print(policy, file=f)
