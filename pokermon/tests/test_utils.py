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

"""Test Utilities"""

import gym

from pokermon.envs import holdem
from pokermon.agents.smoothUCT import smoothUCT
from pokermon.envs.api import PokerGymSimulator as Simulator

def test():
    env = gym.make('TexasHoldem-v1') # holdem.TexasHoldemEnv(2)
    iters = 500001

    simulator = Simulator(env)

    Agent = smoothUCT.smoothUCT(simulator, select_mode='UCT', 
                                save_policy_step=iters-1,
                                rollout_policy_mode='random', C=1.75)

    Agent.search(iters)

test()
