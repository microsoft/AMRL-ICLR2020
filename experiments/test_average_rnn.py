import tensorflow as tf
import numpy as np
from ray.rllib.models.average_rnn import AverageRNNCell
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

import unittest

class RNNTest(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        # batch_sz = 2
        # input_sz = 3
        c  = tf.constant([[0, 0, 0],
                          [0, 0, 0]], dtype=tf.float32) # [batch_sz, in_sz+1]
        h = tf.constant([[0, 0, 0],
                         [0, 0, 0]], dtype=tf.float32) # [batch_sz, in_sz+1]
        self.st = LSTMStateTuple(c, h)
        # [batch_sz, in_sz]:
        self.test_in = tf.constant([[1, 2],
                                    [2, 3]], dtype=tf.float32)
        # [batch_sz, traj_len, in_sz]:
        self.test_seq = tf.constant([[[1, 2], [2, 5], [3, 2], [-1, 0]], # seq1
                                     [[2, 3], [7, 7], [3, 2], [0.5, 1]]], dtype=tf.float32) # seq2

    def runRNN(self, rnn):
        return tf.nn.dynamic_rnn(
                rnn,
                self.test_seq,
                initial_state=self.st,
                time_major=False,
                dtype=tf.float32)


    def testMax(self):
        # Test call function
        rnn = AverageRNNCell(2, max_instead=True)
        output, state = rnn.call(self.test_in, self.st)
        correct_out = np.array([[1,2,1], [2,3,1]]) # Just add time to each item in batch
        self.assertTrue((self.sess.run(output) == correct_out).all())
        self.assertTrue((self.sess.run(state.c) == correct_out).all()) # After 1 step, average output (h) is same as sum (c)
        self.assertTrue((self.sess.run(state.h) == correct_out).all())
        # Test on test sequence
        rnn_out, rnn_state = self.runRNN(rnn)
        correct_out = np.array(
            [[[[1 ,  2,   1   ], # seq1, count on right side
               [2,   5,   2   ],
               [3,   5,   3   ],
               [3,   5,   4   ]],

              [[2,     3,   1   ], # seq2
               [7,     7,   2   ],
               [7,     7,   3   ],
               [7,     7,   4   ]]]])

        correct_st_h = np.array(
            [[3, 5, 4],
             [7, 7, 4]])

        correct_st_c = np.array(
            [[3, 5, 4],
             [7, 7, 4]])

        # assert False, self.sess.run(rnn_out)
        self.assertTrue((self.sess.run(rnn_out) == correct_out).all())
        self.assertTrue((self.sess.run(rnn_state.h) == correct_st_h).all())
        self.assertTrue((self.sess.run(rnn_state.c) == correct_st_c).all())

    def testAverage(self):
        # Test call function
        rnn = AverageRNNCell(2)
        output, state = rnn.call(self.test_in, self.st)
        correct_out = np.array([[1,2,1], [2,3,1]]) # Just add time to each item in batch
        self.assertTrue((self.sess.run(output) == correct_out).all())
        self.assertTrue((self.sess.run(state.c) == correct_out).all()) # After 1 step, average output (h) is same as sum (c)
        self.assertTrue((self.sess.run(state.h) == correct_out).all())
        # Test on test sequence
        rnn_out, rnn_state = self.runRNN(rnn)
        correct_out = np.array(
            [[[[1.,    2.,    1.   ], # seq1, count on right side
               [3/2,   7/2,   2.   ],
               [6/3,   9/3,   3.   ],
               [5/4,   9/4,   4.   ]],

              [[2.,    3.,    1.   ], # seq2
               [9/2,   10/2,  2.   ],
               [12/3,  12/3,  3.   ],
               [12.5/4,13/4,  4.   ]]]])

        correct_st_h = np.array( # final h
            [[5/4,   9/4,   4.   ],
             [12.5/4,13/4,  4.   ]])

        correct_st_c = np.array( # final c
            [[5,    9,   4.   ],
             [12.5, 13,  4.   ]])

        # assert False, self.sess.run(rnn_out)
        self.assertTrue((self.sess.run(rnn_out) == correct_out).all())
        self.assertTrue((self.sess.run(rnn_state.h) == correct_st_h).all())
        self.assertTrue((self.sess.run(rnn_state.c) == correct_st_c).all())

    def testNoCount(self):
        # Test call function
        rnn = AverageRNNCell(2, count=False)
        output, state = rnn.call(self.test_in, self.st)
        correct_out = np.array([[1,2], [2,3]]) # Just add time to each item in batch
        correct_c = np.array([[1,2,1], [2,3,1]]) # Just add time to each item in batch
        correct_h = np.array([[1,2,1], [2,3,1]]) # Just add time to each item in batch
        self.assertTrue((self.sess.run(output) == correct_out).all())
        self.assertTrue((self.sess.run(state.c) == correct_c).all()) # After 1 step, average output (h) is same as sum (c)
        self.assertTrue((self.sess.run(state.h) == correct_h).all())
        # Test on test sequence
        rnn_out, rnn_state = self.runRNN(rnn)
        correct_out = np.array(
            [[[[1.,    2.   ], # seq1, count on right side
               [3/2,   7/2  ],
               [6/3,   9/3  ],
               [5/4,   9/4  ]],

              [[2.,    3.   ], # seq2
               [9/2,   10/2 ],
               [12/3,  12/3 ],
               [12.5/4,13/4 ]]]])

        correct_st_h = np.array( # final h
            [[5/4,   9/4,    4. ],
             [12.5/4,13/4,   4. ]])

        correct_st_c = np.array( # final c
            [[5,    9,   4.   ],
             [12.5, 13,  4.   ]])

        # assert False, self.sess.run(rnn_out)
        self.assertTrue((self.sess.run(rnn_out) == correct_out).all())
        self.assertTrue((self.sess.run(rnn_state.h) == correct_st_h).all())
        self.assertTrue((self.sess.run(rnn_state.c) == correct_st_c).all())

    def testSum(self):
        # Test call function
        rnn = AverageRNNCell(2, sum_instead=True)
        output, state = rnn.call(self.test_in, self.st)
        correct_out = np.array([[1,2,1], [2,3,1]]) # Just add time to each item in batch
        self.assertTrue((self.sess.run(output) == correct_out).all())
        self.assertTrue((self.sess.run(state.c) == correct_out).all()) # After 1 step, average output (h) is same as sum (c)
        self.assertTrue((self.sess.run(state.h) == correct_out).all())
        # Test on test sequence
        rnn_out, rnn_state = self.runRNN(rnn)
        correct_out = np.array(
            [[[[1.,  2.,  1.   ], # seq1, count on right side
               [3,   7,   2.   ],
               [6,   9,   3.   ],
               [5,   9,   4.   ]],

              [[2.,    3.,  1.   ], # seq2
               [9,     10,  2.   ],
               [12,    12,  3.   ],
               [12.5,  13,  4.   ]]]])

        correct_st_h = np.array(
            [[5,    9,   4.   ],
             [12.5, 13,  4.   ]])

        correct_st_c = np.array(
            [[5,    9,   4.   ],
             [12.5, 13,  4.   ]])

        # assert False, self.sess.run(rnn_out)
        self.assertTrue((self.sess.run(rnn_out) == correct_out).all())
        self.assertTrue((self.sess.run(rnn_state.h) == correct_st_h).all())
        self.assertTrue((self.sess.run(rnn_state.c) == correct_st_c).all())

    def testStraightThrough(self):
        def assert_same_output_and_grad(rnn, output_rnn, grad_rnn):
            """ make sure rnn has same output as output_rnn but same gradient as grad_rnn) """
            rnn_out, _ = self.runRNN(rnn)
            grad_rnn_out, _     = self.runRNN(grad_rnn)
            output_rnn_out, _   = self.runRNN(output_rnn)
            self.assertTrue((self.sess.run(rnn_out) == self.sess.run(output_rnn_out)).all()) # same output
            rnn_grad = tf.gradients(rnn_out, self.test_seq)
            grad_rnn_grad  = tf.gradients(grad_rnn_out, self.test_seq)
            rnn_grad_array = np.array(self.sess.run(rnn_grad))
            grad_rnn_grad_array = np.array(self.sess.run(grad_rnn_grad))
            self.assertTrue((rnn_grad_array == grad_rnn_grad_array).all()) # same grad
        through_rnn = AverageRNNCell(2, straight_through=True)
        sum_rnn     = AverageRNNCell(2, sum_instead=True)
        avg_rnn     = AverageRNNCell(2)
        assert_same_output_and_grad(through_rnn, avg_rnn, sum_rnn)
        max_through_rnn = AverageRNNCell(2, straight_through=True, max_instead=True)
        max_rnn = AverageRNNCell(2, max_instead=True)
        assert_same_output_and_grad(max_through_rnn, max_rnn, sum_rnn)

if __name__ == "__main__":
    unittest.main(verbosity=2)