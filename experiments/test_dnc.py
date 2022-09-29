import tensorflow as tf
import numpy as np
from dnc.dnc import DNC, DNCState
from ray.rllib.models.lstm import dnc_state_2_vector, vector_2_dnc_state, dnc_state_from_dict, tensors_from_dnc_state

import unittest

def dnc_np_states_equal(state1, state2, print_diff=False):
    assert type(state1) is DNCState, type(state1)
    assert type(state2) is DNCState, type(state2)
    tensors_for_state1 = tensors_from_dnc_state(state1)
    tensors_for_state2 = tensors_from_dnc_state(state2)
    for tensor1, tensor2 in zip(tensors_for_state1, tensors_for_state2):
        if tensor1.shape != tensor2.shape:
            if print_diff:
                print("\n\ntensor1:\n", tensor1)
                print("\n\ntensor2:\n", tensor2)
            return False
        if not ((tensor1 == tensor2).all()):
            if print_diff:
                print("\n\ntensor1:\n", tensor1)
                print("\n\ntensor2:\n", tensor2)
                print("\n\ntensor1-tensor2:\n", tensor1-tensor2)
            return False
    return True

class RNNTest(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.access_config = {
          "memory_size": 13,
          "word_size": 14,
          "num_reads": 5,
          "num_writes": 2,
        }
        self.controller_config = {
          "hidden_size": 63,
        }
        self.clip_value = 21
        self.out_size = 257

    def testVectorConversion(self):
        # Convert to vector and back and make sure equal
        for batch_sz in [1, 5, 100]:
            # Test with initial state (all 0's)
            dnc = DNC(self.access_config, self.controller_config, self.out_size, self.clip_value)
            state = dnc.initial_state(batch_sz)
            state_vec =  dnc_state_2_vector(state)
            self.assertTrue((self.sess.run(state_vec) == 0).all()) # Make sure init state is all 0s
            self.assertEqual(len(state_vec.shape.as_list()), 2)
            reconstructed_state = vector_2_dnc_state(dnc, state_vec)
            self.assertTrue(dnc_np_states_equal(self.sess.run(state), self.sess.run(reconstructed_state)))

            # Test with random state
            random_dnc_state_dict = {}
            for item, attr in [(state.controller_state, "hidden"),
                               (state.controller_state, "cell"),
                               (state.access_state.linkage, "link"),
                               (state.access_state.linkage, "precedence_weights"),
                               (state.access_state, "memory"),
                               (state.access_state, "read_weights"),
                               (state.access_state, "write_weights"),
                               (state.access_state, "usage"),
                               (state, "access_output")
                              ]:
                rand_tensor = tf.random_uniform(getattr(item, attr).shape)
                random_dnc_state_dict[attr] = rand_tensor
            rand_state = dnc_state_from_dict(random_dnc_state_dict, state)
            self.assertFalse(dnc_np_states_equal(self.sess.run(rand_state), self.sess.run(state)))
            rand_state_vec =  dnc_state_2_vector(rand_state)
            self.assertEqual(len(rand_state_vec.shape.as_list()), 2)
            rand_reconstructed_state = vector_2_dnc_state(dnc, rand_state_vec)
            # make sure to eval at same time, since they are randomized
            rand_reconstructed_state_np, rand_state_np = self.sess.run([rand_reconstructed_state, rand_state])
            self.assertTrue(dnc_np_states_equal(rand_reconstructed_state_np, rand_state_np))
            


if __name__ == "__main__":
    unittest.main(verbosity=2)