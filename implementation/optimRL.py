#%% 
#region Load Modules
from distutils.log import debug
import itertools
import os.path
import json
import plotly.express as px
from collections import OrderedDict

import pandas as pd
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import tensorflow as tf
#endregion

#region Load & Prepare Data
# load data
my_path = os.getcwd()
data_f = open(os.path.join(my_path, 'data','simple.json'))
data = json.load(data_f)
data_f.close()

# prepare data
N = [i for i in range(int(data['index']['n']))]
P = data['index']['p']

tg = {p:int(data['target'][p]) for p in P}
dm = {p:int(data['demand'][p]) for p in P}
cap = int(data['capacity'])
scap = int(data['surge_capacity'])
oc = {p:int(data['params']['oc'][p]) for p in P}
lb = {p:int(data['params']['lb'][p]) for p in P}
gam = float(data['params']['disc'])

# expected values
E_x = [ (cap *(0.95**i)) for i in N]
E_y = {p:dm[p] for p in P}

# booking cost
book = {(p,n):0 for p in P for n in N}
for p,n in itertools.product(P, N):
    if (n+1) > tg[p]:
        for k in range(0, (n+1) - tg[p]):
            book[(p,n)] += (gam**k) * lb[p]
#endregion
#region Environment
class SchedEnv(gym.Env):

    def __init__(self, N, P, cap, scap, dm, tg, oc, lb, gam, debug=False, seed=1239) -> None:
        super().__init__()

        # Metadata
        self.debug= debug
        self.random_stream = np.random.RandomState(seed = seed)

        # Initialize Data
        self.N = N
        self.P = P
        self.cap = cap
        self.scap = scap
        self.dm = dm
        self.tg = tg
        self.oc = oc
        self.lb = lb
        self.gam = gam
        self.tot_steps = 50
        self.cur_step = 1
        self.cost = 0


        # State Space
        high = np.array([[self.dm[p]*5 for p in self.P]])
        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=0, high=cap, shape = (1,len(self.N)), dtype=np.int64),
            "y": spaces.Box(low=0, high=high, shape = (1,len(self.P)), dtype=np.int64) 
        })

        # Action Space
        low = [0 for i in range(len(N)+1)]
        high = [scap] + [cap for n in N]
        low = np.array([low for p in P])
        high = np.array([high for p in P])
        self.action_space = spaces.flatten_space( spaces.Box(low=low, high=high, dtype=np.int64) )

        # Initialize State
        self.state = None

    def reset(self, debug=False):
        self.debug = debug
        st_x = np.array([[min(self.random_stream.poisson(cap*(0.9)**n),self.cap) for n in self.N]], dtype=np.int64)
        st_y = np.array([[min(self.random_stream.poisson(self.dm[p]),self.dm[p]*5) for p in self.P]], dtype=np.int64)
        self.state = {'x': st_x, 'y': st_y}
        self.cur_step = 1

        if self.debug == True:
            print('\tReset State')
            print(f"x: {self.state['x']}")
            print(f"y: {self.state['y']}")

        err_msg = f"{self.state!r} ({type(self.state)}) invalid"
        assert self.observation_space.contains(self.state), err_msg
        
        return(self.state)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        # init state
        if self.debug == True:
            print('\tInit State')
            print(f"x: {self.state['x'][0]}")
            print(f"y: {self.state['y'][0]}")

        # action
        action = action.reshape(len(self.P), len(self.N) + 1)

        self.ac_z = np.round(action[:,0])
        self.ac_a = np.round(action[:,1:])
        if self.debug == True:
            print('\tAction')
            print(f"a: {self.ac_a}")
            print(f"z: {self.ac_z}")

        # Execute Action
        cost = 0

        # Divert Patients
        rem_cap = self.scap
        for p in range(len(self.P)):
            action = min(self.ac_z[p], self.state['y'][0][p], rem_cap)
            cost += self.oc[P[p]] * action
            rem_cap -= action
            self.state['y'][0][p] -= action

        if self.debug == True:
            print('\tPost Divert State')
            print(f"x: {self.state['x'][0]}")
            print(f"y: {self.state['y'][0]}")
            print(f"cost: {cost}")

        # Schedule remaining patients
        for p in range(len(self.P)):
            for n in self.N:
                rem_cap = self.cap - self.state['x'][0][n]
                action = min(self.ac_a[p,n], self.state['y'][0][p], rem_cap)
                self.state['x'][0][n] += action
                self.state['y'][0][p] -= action
                cost += action * book[(P[p],n)]

        # Penalize those who are waiting longer
        for p in range(len(self.P)):
            cost += self.state['y'][0][p] * self.lb[P[p]]

        if self.debug == True:
            print('\tPost Schedule State')
            print(f"x: {self.state['x'][0]}")
            print(f"y: {self.state['y'][0]}")
            print(f"cost: {cost}")
        
        # Simulate New Demand
        for p in range(len(self.P)):
            self.state['y'][0][p] += self.random_stream.poisson(self.dm[P[p]])

        
        # Move everyone over
        for n in N:
            if n != N[-1]: self.state['x'][0][n] = self.state['x'][0][n+1]
            else: self.state['x'][0][n] = 0

        if self.debug == True:
            print('\tPost Transition State')
            print(f"x: {self.state['x'][0]}")
            print(f"y: {self.state['y'][0]}")
            print(f"cost: {cost}")

        # print(cost)

        self.cur_step += 1
        done = False
        if self.cur_step >= self.tot_steps:
            done = True

        return self.state, -cost, done, {}
#endregion
#%%
seed = 12342345
env = SchedEnv(N, P, cap, scap, dm, tg, oc, lb, gam, seed=seed)

#%% Compare Models
model = A2C("MultiInputPolicy", env, verbose=1, seed=seed, device='cuda', tensorboard_log='data/PPO_RL')
model.learn(total_timesteps=5000)
evaluate_policy(model, env)

#%%
# obs = env.reset(debug=True)
# for i in range(10):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     # env.render()
#     if done:
#       obs = env.reset()

