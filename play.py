from time import sleep

import gym
import gym_watten

from baselines import deepq


def main():
    env = gym.make("Watten-v0")
    act = deepq.load("cartpole_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act([obs])[0])
            episode_rew += rew
            sleep(1)

        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()