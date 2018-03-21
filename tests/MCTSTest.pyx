import unittest
from gym_watten.envs.watten_env cimport WattenEnv, Observation
from src.MCTS cimport MCTS, Storage, MCTSState
from src.LookUp cimport LookUp, ModelOutput

class MCTSTest(unittest.TestCase):

    def test_add_state(self):
        """ Memorize """
        cdef WattenEnv env = WattenEnv()
        cdef LookUp model = LookUp()
        cdef Storage storage
        cdef MCTS mcts = MCTS()
        cdef MCTSState root

        mcts.add_state(&root, 0.5, env, -1)

        self.assertEqual(root.childs.size(), 1, "No child state created")
        self.assertEqual(root.childs[0].p, 0.5, "Wrong p")
        self.assertEqual(root.childs[0].n, 0, "Wrong n")
        self.assertEqual(root.childs[0].w, 0, "Wrong w")
        self.assertEqual(root.childs[0].v, 0, "Wrong v")
        self.assertEqual(root.childs[0].current_player, env.current_player, "Wrong current_player")
        self.assertEqual(root.childs[0].end_v, -1, "Wrong end_v")
        self.assertEqual(root.childs[0].is_root, False, "Wrong is_root")

    def test_is_leaf_node(self):
        """ Memorize """
        cdef MCTS mcts = MCTS()
        cdef MCTSState root
        cdef MCTSState child
        root.childs.push_back(child)

        self.assertFalse(mcts.is_state_leaf_node(&root), "Root is no leaf node")
        self.assertTrue(mcts.is_state_leaf_node(&child), "Child is a leaf node")

    def test_calc_q_leaf(self):
        cdef MCTS mcts = MCTS()
        cdef MCTSState root
        root.end_v = 1

        cdef int n
        cdef float q = mcts.calc_q(&root, 0, &n)

        self.assertEqual(q, 1, "Wrong q")
        self.assertEqual(n, 1, "Wrong n")
        self.assertEqual(mcts.calc_q(&root, 1, &n), -1, "Wrong q (other player)")

    def test_calc_q_same_player(self):
        cdef MCTS mcts = MCTS()
        cdef MCTSState root
        root.end_v = 0
        root.current_player = 0
        root.n = 0

        cdef int n
        cdef float q = mcts.calc_q(&root, 0, &n)

        self.assertEqual(q, 0, "Wrong q")
        self.assertEqual(n, 0, "Wrong n")

        root.current_player = 1
        q = mcts.calc_q(&root, 1, &n)

        self.assertEqual(q, 0, "Wrong q (other player)")
        self.assertEqual(n, 0, "Wrong n (other player)")

        root.current_player = 0
        root.n = 18
        root.w = 9
        q = mcts.calc_q(&root, 0, &n)

        self.assertEqual(q, 0.5, "Wrong q (n != 0)")
        self.assertEqual(n, 18, "Wrong n (n != 0)")

    def test_calc_q_other_player(self):
        cdef MCTS mcts = MCTS()
        cdef MCTSState root
        root.end_v = 0
        root.current_player = 0
        root.n = 18
        root.w = 9

        cdef int n
        cdef float q = mcts.calc_q(&root, 1, &n)

        self.assertEqual(q, 0, "Wrong q")
        self.assertEqual(n, 0, "Wrong n")

    def test_calc_q_other_player_one_child(self):
        cdef MCTS mcts = MCTS()
        cdef MCTSState root
        root.end_v = 0
        root.current_player = 1

        cdef MCTSState child
        child.end_v = 0
        child.current_player = 0
        child.n = 18
        child.w = 9
        root.childs.push_back(child)

        cdef int n
        cdef float q = mcts.calc_q(&root, 0, &n)

        self.assertEqual(q, 0.5, "Wrong q")
        self.assertEqual(n, 18, "Wrong n")

    def test_calc_q_other_player_two_childs(self):
        cdef MCTS mcts = MCTS()
        cdef MCTSState root
        root.end_v = 0
        root.current_player = 1

        cdef MCTSState child
        child.end_v = 0
        child.current_player = 0
        child.n = 18
        child.w = 9
        root.childs.push_back(child)

        child.current_player = 1
        child.n = 35
        child.w = -1
        root.childs.push_back(child)

        cdef int n
        cdef float q = mcts.calc_q(&root, 0, &n)

        self.assertEqual(q, 0.5, "Wrong q")
        self.assertEqual(n, 18, "Wrong n")

    def test_calc_q_other_player_two_levels(self):
        cdef MCTS mcts = MCTS()
        cdef MCTSState root
        root.end_v = 0
        root.current_player = 1

        cdef MCTSState child
        child.end_v = 0
        child.current_player = 0
        child.n = 20
        child.w = 10
        root.childs.push_back(child)

        child.current_player = 1
        child.n = 40
        child.w = -1

        cdef MCTSState deeper_child
        deeper_child.end_v = 0
        deeper_child.current_player = 0
        deeper_child.n = 20
        deeper_child.w = -5
        child.childs.push_back(deeper_child)

        deeper_child.current_player = 0
        deeper_child.n = 10
        deeper_child.w = -1
        child.childs.push_back(deeper_child)

        root.childs.push_back(child)

        cdef int n
        cdef float q = mcts.calc_q(&root, 0, &n)

        self.assertAlmostEqual(q, 0.08, 5, "Wrong q")
        self.assertEqual(n, 50, "Wrong n")


    def test_mcts_sample_leaf(self):
        cdef MCTS mcts = MCTS()
        cdef WattenEnv env = WattenEnv()
        env.seed(42)
        cdef Observation obs = env.reset()
        cdef LookUp model = LookUp()

        cdef ModelOutput output
        for i in range(32):
            output.p[i] = 1
        output.p[env.players[0].hand_cards[1].id] = 0.5
        output.v = -0.25
        model.memorize(&obs, &output)

        cdef MCTSState root = mcts.create_root_state(env)

        cdef int player
        cdef float v = mcts.mcts_sample(env, &root, model, &player)

        self.assertEqual(root.childs.size(), 3, "No child created")
        for i in range(3):
            self.assertEqual(root.childs[i].current_player, 1, "Wrong current player, child " + str(i))
        self.assertEqual(root.childs[0].p, 1, "Wrong p, child 0")
        self.assertEqual(root.childs[1].p, 0.5, "Wrong p, child 1")
        self.assertEqual(root.childs[2].p, 1, "Wrong p, child 2")
        self.assertEqual(root.n, 1, "Wrong n")
        self.assertEqual(root.w, -0.25, "Wrong w")
        self.assertEqual(v, -0.25, "Wrong v")
        self.assertEqual(player, 0, "Wrong player")

    def test_mcts_sample_step(self):
        cdef MCTS mcts = MCTS()
        cdef WattenEnv env = WattenEnv()
        env.seed(42)
        cdef Observation obs = env.reset()
        cdef LookUp model = LookUp()
        cdef MCTSState root = mcts.create_root_state(env)
        root.n = 1
        root.w = -0.25

        obs = env.step(3)
        mcts.add_state(&root, 1, env)

        cdef ModelOutput output
        for i in range(32):
            output.p[i] = 1
        output.p[env.players[1].hand_cards[0].id] = 0.75
        output.v = 0.5
        model.memorize(&obs, &output)

        cdef MCTSState* child = &root.childs[0]

        cdef int player
        cdef float v = mcts.mcts_sample(env, &root, model, &player)

        self.assertEqual(child.childs.size(), 3, "No child created")
        self.assertEqual(child.childs[0].current_player, 0, "Wrong current player, child 0")
        self.assertEqual(child.childs[1].current_player, 1, "Wrong current player, child 1")
        self.assertEqual(child.childs[2].current_player, 1, "Wrong current player, child 2")
        self.assertEqual(child.childs[0].p, 0.75, "Wrong p, child 0")
        self.assertEqual(child.childs[1].p, 1, "Wrong p, child 1")
        self.assertEqual(child.childs[2].p, 1, "Wrong p, child 2")
        self.assertEqual(root.n, 1, "Wrong n")
        self.assertEqual(root.w, -0.25, "Wrong w")
        self.assertEqual(v, 0.5, "Wrong v")
        self.assertEqual(player, 1, "Wrong player")

    def test_mcts_sample_end_v(self):
        cdef MCTS mcts = MCTS()
        cdef WattenEnv env = WattenEnv()
        cdef Observation obs = env.reset()
        cdef LookUp model = LookUp()

        env.seed(42)
        env.reset()
        env.step(3)
        env.step(0)

        env.step(5)
        env.step(4)

        env.step(7)
        cdef MCTSState root = mcts.create_root_state(env)
        root.n = 1
        root.w = -0.25

        env.step(1)
        mcts.add_state(&root, 1, env, (1 if env.last_winner is 0 else -1))

        cdef MCTSState* child = &root.childs[0]

        cdef int player
        cdef float v = mcts.mcts_sample(env, &root, model, &player)

        self.assertEqual(child.childs.size(), 0, "Child created")
        self.assertEqual(root.n, 2, "Wrong n")
        self.assertEqual(root.w, -1.25, "Wrong w")
        self.assertEqual(child.n, 1, "Wrong n")
        self.assertEqual(child.w, 1, "Wrong w")
        self.assertEqual(v, 1, "Wrong v")
        self.assertEqual(player, -1, "Wrong player")


    def test_memorize(self):
        """ Memorize """
        cdef WattenEnv env = WattenEnv()
        cdef LookUp model = LookUp()
        cdef Storage storage
        cdef MCTS mcts = MCTS()

        #mcts.mcts_generate(env, model, &storage)

