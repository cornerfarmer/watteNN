from gym_watten.envs.watten_env cimport WattenEnv, Observation
from src.LookUp cimport LookUp, ModelOutput
from src.ModelRating cimport ModelRating
from libcpp.vector cimport vector
from libcpp cimport bool
from time import sleep
from libc.stdlib cimport rand

cdef class Game:

    def __cinit__(self, WattenEnv env):
        self.env = env

    cpdef int match(self, LookUp agent1, LookUp agent2, bool render=False, bool reset=True):
        cdef Observation obs
        cdef ModelOutput output
        cdef int a
        if reset:
            self.env.reset(&obs)
        else:
            self.env.regenerate_obs(&obs)

        while not self.env.is_done():
            if self.env.current_player == 0:
                agent1.predict_single(&obs, &output)
                a = agent1.valid_step(output.p, &self.env.players[self.env.current_player].hand_cards)
            else:
                agent2.predict_single(&obs, &output)
                a = agent2.valid_step(output.p, &self.env.players[self.env.current_player].hand_cards)

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

    cpdef float compare_given_games(self, LookUp agent1, LookUp agent2, ModelRating rating):
        cdef int i, first_player_wins, winner, start_player
        first_player_wins = 0

        for i in range(rating.eval_games.size()):
            for start_player in range(2):
                self.env.set_state(&rating.eval_games[i])
                self.env.current_player = start_player
                winner = self.match(agent1, agent2, render=False, reset=False)
                first_player_wins += (winner == 0)

        return <float>first_player_wins / (rating.eval_games.size() * 2)

    def test(self):
        pass