from gym_watten.envs.watten_env cimport WattenEnv, Observation
from src.LookUp cimport LookUp, ModelOutput
from libcpp.vector cimport vector
from time import sleep
from libc.stdlib cimport rand

cdef class Game:

    def __cinit__(self, WattenEnv env):
        self.env = env

    cdef int match(self, LookUp agent1, LookUp agent2, render=False, reset=True):
        cdef Observation obs
        cdef LookUp current_agent
        cdef ModelOutput output
        if reset:
            self.env.reset(&obs)
        else:
            self.env.regenerate_obs(&obs)

        while not self.env.is_done():
            current_agent = agent1 if self.env.current_player == 0 else agent2
            current_agent.predict_single(&obs, &output)

            a = current_agent.valid_step(output.p, &self.env.players[self.env.current_player].hand_cards)

            self.env.step(a, &obs)

            if render:
                self.env.render('human')
                sleep(2)
           # if env.lastTrick  is not None:
            #    break

        return self.env.last_winner

    """cdef float compare(self, LookUp agent1, LookUp agent2):
        cdef vector[LookUp] agents
        agents.push_back(agent1)
        agents.push_back(agent2)
        first_player_wins = 0

        for i in range(10000):
            start_player = rand() % 2
            winner = self.match([agents[start_player], agents[1 - start_player]])
            first_player_wins += ((winner == 0) == (start_player == 0))
            #print(start_player, winner)

        return first_player_wins / 10000"""

    cdef float compare_given_games(self, LookUp agent1, LookUp agent2, vector[State]* games):
        first_player_wins = 0

        for i in range(games.size()):
            for start_player in range(2):
                self.env.set_state(&games[0][i])
                self.env.current_player = start_player
                winner = self.match(agent1, agent2, render=False, reset=False)
                first_player_wins += (winner == 0)

        return first_player_wins / (games.size() * 2)