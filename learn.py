import time

import gym
import gym_watten
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule


def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    with U.make_session(8):
        # Create the environment
        env = gym.make("Watten-v0")
        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(env.observation_space.shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=1e-3),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=20000, initial_p=1.0, final_p=0)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        train_writer = tf.summary.FileWriter('tensorboard/' + time.strftime("%Y%m%d-%H%M%S"))

        episode_rewards = [0.0]
        obs = env.reset()
        storeRewardNextRound = True
        for t in itertools.count():
            # Take action and update exploration to the newest value
            action = act([obs], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)

            if not storeRewardNextRound or len(rew) > 0 and rew[0] < 0:

                if not rew[0] < 0:
                    replay_buffer.add(last_obs, last_action, rew[1], last_new_obs, float(last_done))
                    episode_rewards[-1] += rew[1]

                replay_buffer.add(obs, action, rew[0], new_obs, float(done))
                storeRewardNextRound = True

                episode_rewards[-1] += rew[0]
            else:
                storeRewardNextRound = False
                last_obs, last_action, last_new_obs, last_done = obs, action, new_obs, done

            obs = new_obs

            if done:
                obs = env.reset()
                episode_rewards.append(0)

            is_solved = t > 60000
            if is_solved:
                # Show off the result
                env.render()
                time.sleep(1)
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            if done and len(episode_rewards) % 100 == 0:

                g = 0
                avg_reward = 0
                starting_player = 0
                while g < 100:
                    if env.current_player == starting_player:
                        action = None
                    else:
                        action = act([obs], update_eps=0)[0]
                    obs, rew, done, _ = env.step(action)

                    if len(rew) > 0 and rew[0] < 0:
                        if env.current_player == starting_player:
                            avg_reward += rew[0]
                    elif len(rew) == 2:
                        avg_reward += rew[starting_player]

                    if done:
                        starting_player = 0 if g < 50 else 1
                        obs = env.reset()
                        g += 1
                obs = env.reset()

                summary = tf.Summary()
                summary.value.add(tag="mean episode reward", simple_value=round(np.mean(episode_rewards[-101:-1]), 1))
                summary.value.add(tag="rating", simple_value=avg_reward / 100)
                summary.value.add(tag="exploring", simple_value=int(100 * exploration.value(t)))
                train_writer.add_summary(summary, t)
                train_writer.flush()