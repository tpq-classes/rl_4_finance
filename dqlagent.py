#
# Deep Q-Learning Agent
#
# (c) Dr. Yves J. Hilpisch
# Reinforcement Learning for Finance
#

import os
import random
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

warnings.simplefilter('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

opt = keras.optimizers.Adam


class DQLAgent:
    def __init__(self, symbol, feature, n_features, env, hu=24, lr=0.001):
        self.epsilon = 1.0
        self.epsilon_decay = 0.9975
        self.epsilon_min = 0.1
        self.memory = list()
        self.batch_size = 128
        self.gamma = 0.5
        self.trewards = deque(maxlen=2000)
        self.max_treward = -np.inf
        self.n_features = n_features
        self.env = env
        self.episodes = 0
        self._create_model(hu, lr)

    def _create_model(self, hu, lr):
        self.model = Sequential()
        self.model.add(Dense(hu, activation='relu',
                             input_dim=self.n_features))
        self.model.add(Dense(hu, activation='relu'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=opt(learning_rate=lr))

    def _reshape(self, state):
        state = state.flatten()
        return np.reshape(state, [1, len(state)])

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        #return np.argmax(self.model.predict(state)[0])

        q = self.model(tf.convert_to_tensor(state, dtype=tf.float32), training=False)
        return int(tf.argmax(q[0]).numpy())

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)


        #for state, action, next_state, reward, done in batch:
        #    if not done:
        #        reward += self.gamma * np.amax(
        #            self.model.predict(next_state)[0])
        #        target = self.model.predict(state)
        #        target[0, action] = reward
        #        self.model.fit(state, target, epochs=1, verbose=False)

        states      = np.vstack([b[0] for b in batch]).astype(np.float32)
        actions     = np.array([b[1] for b in batch], dtype=np.int32)
        next_states = np.vstack([b[2] for b in batch]).astype(np.float32)
        rewards     = np.array([b[3] for b in batch], dtype=np.float32)
        dones       = np.array([b[4] for b in batch], dtype=np.bool_)

        # Q(s, :)
        q_states = self.model(states, training=False).numpy()          # (B, A)

        # max_a' Q(s', a')
        q_next = self.model(next_states, training=False).numpy()       # (B, A)
        max_q_next = np.max(q_next, axis=1)                            # (B,)

        targets = q_states.copy()
        targets[np.arange(self.batch_size), actions] = rewards + self.gamma * max_q_next * (~dones)

         # One update only (FAST)
        self.model.train_on_batch(states, targets)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes):
        for e in range(1, episodes + 1):
            self.episodes += 1
            state, _ = self.env.reset()
            state = self._reshape(state)
            treward = 0
            for f in range(1, 5000):
                action = self.act(state)
                next_state, reward, done, trunc, _ = self.env.step(action)
                treward += reward
                next_state = self._reshape(next_state)
                self.memory.append(
                    [state, action, next_state, reward, done])
                state = next_state 
                if done:
                    self.trewards.append(treward)
                    self.max_treward = max(self.max_treward, treward)
                    templ = f'episode={e:4d} | treward={treward:7.2f}'
                    templ += f' | max={self.max_treward:7.2f}'
                    print(templ, end='\r')
                    break
            if len(self.memory) > self.batch_size and (e % 5 == 0):
                self.replay()
        print()
        
    def test(self, episodes, min_accuracy=0.0,
             min_performance=0.0, verbose=True):
        ma = self.env.min_accuracy
        self.env.min_accuracy = min_accuracy
        if hasattr(self.env, 'min_performance'):
            mp = self.env.min_performance
            self.env.min_performance = min_performance
            self.performances = list()
        for e in range(1, episodes + 1):
            state, _ = self.env.reset()
            state = self._reshape(state)
            for f in range(1, 5001):
                # action = np.argmax(self.model.predict(state)[0])
                q = self.model(tf.convert_to_tensor(state, dtype=tf.float32), training=False)
                action = int(tf.argmax(q[0]).numpy())
                state, reward, done, trunc, _ = self.env.step(action)
                state = self._reshape(state)
                if done:
                    templ = f'total reward={f:4d} | '
                    templ += f'accuracy={self.env.accuracy:.3f}'
                    if hasattr(self.env, 'min_performance'):
                        self.performances.append(self.env.performance)
                        templ += f' | performance={self.env.performance:.3f}'
                    if verbose:
                        print(templ, end='\n')
                    break
        self.env.min_accuracy = ma
        if hasattr(self.env, 'min_performance'):
            self.env.min_performance = mp

