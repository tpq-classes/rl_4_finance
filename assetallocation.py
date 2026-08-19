#
# Investing Environment and Agent
# Three Asset Case
#
# (c) Dr. Yves J. Hilpisch
# Reinforcement Learning for Finance
#

import os
import math
import random
import numpy as np
import pandas as pd
from scipy import stats
from pylab import plt, mpl
from numpy.random import default_rng
from scipy.optimize import minimize

from dqlagent import *

plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(suppress=True)

opt = keras.optimizers.Adam

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

rng = default_rng(100)

class observation_space:
    def __init__(self, n):
        self.shape = (n,)

class action_space:
    def __init__(self, n):
        self.n = n
    def seed(self, seed):
        random.seed(seed)
    def sample(self):
        rn = rng.random(3)
        return rn / rn.sum()

class Investing:
    # NEW SPEED OPTIMIZATION: store portfolio rows in a list and materialize the DataFrame only when it is read.
    @property
    def portfolios(self):
        return pd.DataFrame(getattr(self, '_portfolio_records', []))

    # NEW SPEED OPTIMIZATION: keep compatibility with existing code that assigns a DataFrame to portfolios.
    @portfolios.setter
    def portfolios(self, value):
        if isinstance(value, pd.DataFrame) and not value.empty:
            self._portfolio_records = value.to_dict('records')
        else:
            self._portfolio_records = []

    def __init__(self, asset_one, asset_two, asset_three,
                 steps=252, amount=1):
        self.asset_one = asset_one
        self.asset_two = asset_two
        self.asset_three = asset_three
        self.steps = steps
        self.initial_balance = amount
        self.portfolio_value = amount
        self.portfolio_value_new = amount
        self.observation_space = observation_space(4)
        self.osn = self.observation_space.shape[0]
        self.action_space = action_space(3)
        self.retrieved = 0
        self._generate_data()
        self.portfolios = pd.DataFrame()
        self.episode = 0

    def _generate_data(self):
        if self.retrieved:
            pass
        else:
            url = 'https://certificate.tpq.io/rl4finance.csv'
            self.raw = pd.read_csv(url, index_col=0, parse_dates=True).dropna()
            # self.retrieved
            # NEW SPEED OPTIMIZATION: avoid re-reading the remote CSV on every reset.
            self.retrieved = True
        self.data = pd.DataFrame()
        self.data['X'] = self.raw[self.asset_one]
        self.data['Y'] = self.raw[self.asset_two]
        self.data['Z'] = self.raw[self.asset_three]
        s = random.randint(self.steps, len(self.data))
        self.data = self.data.iloc[s-self.steps:s]
        self.data = self.data / self.data.iloc[0]

    def _get_state(self):
        Xt = self.data['X'].iloc[self.bar]
        Yt = self.data['Y'].iloc[self.bar]
        Zt = self.data['Z'].iloc[self.bar]
        date = self.data.index[self.bar]
        return np.array(
            [Xt, Yt, Zt, self.xt, self.yt, self.zt]
            ), {'date': date}
        
    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            
    def reset(self):
        self.xt = 0
        self.yt = 0
        self.zt = 0
        self.bar = 0
        self.treward = 0
        self.portfolio_value = self.initial_balance
        self.portfolio_value_new = self.initial_balance
        self.episode += 1
        # NEW SPEED OPTIMIZATION: maintain rolling reward statistics without rebuilding a DataFrame.
        self._pl_pct_window = []
        self._generate_data()
        self.state, info = self._get_state()
        return self.state, info

    def add_results(self, pl):
        # NEW SPEED OPTIMIZATION: append a plain dict instead of concatenating a one-row DataFrame every step.
        self._portfolio_records.append({
                   'e': self.episode, 'date': self.date, 
                   'xt': self.xt, 'yt': self.yt, 'zt': self.zt,
                   'pv': self.portfolio_value, 'pv_new': self.portfolio_value_new,
                   'p&l[$]': pl, 'p&l[%]': pl / self.portfolio_value_new * 100,
                   'Xt': self.state[0],  'Yt': self.state[1], 'Zt': self.state[2],
                   'Xt_new': self.new_state[0],  'Yt_new': self.new_state[1],
                   'Zt_new': self.new_state[2],
                          })
        # df = pd.DataFrame({
        #            'e': self.episode, 'date': self.date, 
        #            'xt': self.xt, 'yt': self.yt, 'zt': self.zt,
        #            'pv': self.portfolio_value, 'pv_new': self.portfolio_value_new,
        #            'p&l[$]': pl, 'p&l[%]': pl / self.portfolio_value_new * 100,
        #            'Xt': self.state[0],  'Yt': self.state[1], 'Zt': self.state[2],
        #            'Xt_new': self.new_state[0],  'Yt_new': self.new_state[1],
        #            'Zt_new': self.new_state[2],
        #                   }, index=[0])
        # self.portfolios = pd.concat((self.portfolios, df), ignore_index=True)
        
    def step(self, action):
        self.bar += 1
        self.new_state, info = self._get_state()
        self.date = info['date']
        if self.bar == 1:
            self.xt = action[0]
            self.yt = action[1]
            self.zt = action[2]
            pl = 0.
            reward = 0.
            self.add_results(pl)
        else:
            self.portfolio_value_new = (
                self.xt * self.portfolio_value * self.new_state[0] / self.state[0] +
                self.yt * self.portfolio_value * self.new_state[1] / self.state[1] +
                self.zt * self.portfolio_value * self.new_state[2] / self.state[2]
            )
            pl = self.portfolio_value_new - self.portfolio_value
            pen = np.mean((np.array([self.xt, self.yt, self.zt]) - action) ** 2)
            self.xt = action[0]
            self.yt = action[1]
            self.zt = action[2]
            self.add_results(pl)
            # NEW SPEED OPTIMIZATION: update rolling Sharpe inputs from a small list instead of rebuilding rolling DataFrame statistics.
            pl_pct = pl / self.portfolio_value_new * 100
            self._pl_pct_window.append(pl_pct)
            if len(self._pl_pct_window) > 20:
                self._pl_pct_window.pop(0)
            ret = pl_pct / 100 * 252
            vol = np.std(self._pl_pct_window, ddof=1) * math.sqrt(252) if len(self._pl_pct_window) > 1 else 0
            sharpe = ret / vol if vol > 0 else 0
            # ret = self.portfolios['p&l[%]'].iloc[-1] / 100 * 252
            # vol = self.portfolios['p&l[%]'].rolling(
            #     20, min_periods=1).std().iloc[-1] * math.sqrt(252)
            # sharpe = ret / vol
            reward = sharpe - pen
            self.portfolio_value = self.portfolio_value_new
        if self.bar == len(self.data) - 1:
            done = True
        else:
            done = False
        self.state = self.new_state
        return self.state, reward, done, False, {}
        

class InvestingAgent(DQLAgent):
    def _create_model(self, hu, lr):
        self.model = Sequential()
        self.model.add(Dense(hu, input_dim=self.n_features,
                        activation='relu'))
        self.model.add(Dense(hu, activation='relu'))
        self.model.add(Dense(1, activation='linear'))
        self.model.compile(loss='mse',
                optimizer=opt(learning_rate=lr))

    def _get_action_grid(self):
        # NEW SPEED OPTIMIZATION: cache a simplex grid for batched 3-asset action search.
        grid = getattr(self, '_action_grid', None)
        if grid is None:
            points = np.linspace(0, 1, 21, dtype=np.float32)
            grid = np.array(
                [(x, y, 1 - x - y)
                 for x in points
                 for y in points
                 if x + y <= 1],
                dtype=np.float32
            )
            self._action_grid = grid
        return grid
        
    def opt_action(self, state):
        # NEW SPEED OPTIMIZATION: evaluate the 3-asset allocation grid in one batched model call instead of repeated SLSQP/model.predict calls.
        try:
            state = self._reshape(state) if state.ndim == 1 else state.astype(np.float32, copy=True)
            grid = self._get_action_grid()
            states = np.repeat(state, len(grid), axis=0)
            states[:, 3:] = grid
            penalties = np.mean((state[0, 3:] - grid) ** 2, axis=1)
            values = self.model(states, training=False).numpy().reshape(-1) - penalties
            self.action = grid[np.argmax(values)]
        except:
            self.action = getattr(self, 'action', np.ones(3) / 3)
        # bnds = 3 * [(0, 1)]
        # cons = [{'type': 'eq', 'fun': lambda x: x.sum() - 1}]
        # def f(state, x):
        #     s = state.copy()
        #     s[0, 3] = x[0]
        #     s[0, 4] = x[1]
        #     s[0, 5] = x[2]
        #     # s[0, 3:] = x
        #     pen = np.mean((state[0, 3:] - x) ** 2)
        #     return self.model.predict(s)[0, 0] - pen
        # try:
        #     state = self._reshape(state)
        #     self.action = minimize(lambda x: -f(state, x),
        #                            3 * [1 / 3],
        #                            bounds=bnds,
        #                            constraints=cons,
        #                            options={
        #                                'eps': 1e-4,
        #                                 },
        #                            method='SLSQP'
        #                           )['x']
        # except:
        #     self.action = self.action
        return self.action
        
    def act(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        action = self.opt_action(state)
        return action

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        # NEW SPEED OPTIMIZATION: vectorize the 3-asset Bellman target and train once per replay.
        states = np.vstack([b[0] for b in batch]).astype(np.float32)
        next_states = np.vstack([b[2] for b in batch]).astype(np.float32)
        rewards = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.bool_)
        grid = self._get_action_grid()
        candidates = np.repeat(next_states, len(grid), axis=0)
        tiled_grid = np.tile(grid, (len(batch), 1))
        candidates[:, 3:] = tiled_grid
        penalties = np.mean(
            (np.repeat(next_states[:, 3:], len(grid), axis=0) - tiled_grid) ** 2,
            axis=1
        ).reshape(len(batch), len(grid))
        next_q = self.model(candidates, training=False).numpy().reshape(len(batch), len(grid))
        best = np.argmax(next_q - penalties, axis=1)
        next_values = next_q[np.arange(len(batch)), best]
        targets = rewards + self.gamma * next_values * (~dones)
        self.model.train_on_batch(states, targets.reshape(-1, 1))
        # for state, action, next_state, reward, done in batch:
        #     if not done:
        #         action = self.opt_action(next_state)
        #         next_state[0, 3:] = action
        #         reward += self.gamma * self.model.predict(next_state)[0, 0]
        #     reward = np.array([reward])
        #     self.model.fit(state, reward, epochs=1,
        #                    verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def test(self, episodes, verbose=True):
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = self._reshape(state)
            treward = 0
            for _ in range(1, len(self.env.data) + 1):
                action = self.opt_action(state)
                state, reward, done, trunc, _ = self.env.step(action)
                state = self._reshape(state)
                treward += reward
                if done:
                    templ = f'episode={e} | '
                    templ += f'total reward={treward:4.2f}'
                    if verbose:
                        print(templ, end='\r')
                    break
        print()

