import numpy as np
import tensorflow as tf

import model_based.utils as utils


# import IPython
# IPython.embed()


class ModelBasedPolicy(object):

    def __init__(self,
                 env,
                 init_dataset,
                 horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1):
        self._env = env
        if hasattr(env, 'cost_fn'):
            self._cost_fn = env.cost_fn
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._init_dataset = init_dataset
        self._horizon = horizon
        self._num_random_action_selection = num_random_action_selection
        self._nn_layers = nn_layers
        self._learning_rate = 1e-3

        self._sess, self._state_ph, self._action_ph, self._next_state_ph, self._reward_ph, \
            self._next_state_pred, self._loss, self._optimizer, self._best_action, \
            self._reward_pred, self._reward_loss, self._reward_optimizer = self._setup_graph()

    def _setup_placeholders(self):
        """
            Creates the placeholders used for training, prediction, and action selection

            returns:
                state_ph: current state
                action_ph: current_action
                next_state_ph: next state

            implementation details:
                (a) the placeholders should have 2 dimensions,
                    in which the 1st dimension is variable length (i.e., None)
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        state_ph = tf.placeholder(tf.float32, (None, self._state_dim), name='state_ph')
        action_ph = tf.placeholder(tf.float32, (None, self._action_dim), name='action_ph')
        next_state_ph = tf.placeholder(tf.float32, (None, self._state_dim), name='next_state_ph')
        reward_ph = tf.placeholder(tf.float32, (None, 1), name='reward_ph')

        return state_ph, action_ph, next_state_ph, reward_ph

    def _dynamics_func(self, state, action, reuse=False):
        """
            Takes as input a state and action, and predicts the next state

            returns:
                next_state_pred: predicted next state

            implementation details (in order):
                (a) Normalize both the state and action by using the statistics of self._init_dataset and
                    the utils.normalize function
                (b) Concatenate the normalized state and action
                (c) Pass the concatenated, normalized state-action tensor through a neural network with
                    self._nn_layers number of layers using the function utils.build_mlp. The resulting output
                    is the normalized predicted difference between the next state and the current state
                (d) Unnormalize the delta state prediction, and add it to the current state in order to produce
                    the predicted next state

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        norm_state = utils.normalize(state,
                                     self._init_dataset.state_mean,
                                     self._init_dataset.state_std)
        norm_action = utils.normalize(action,
                                      self._init_dataset.action_mean,
                                      self._init_dataset.action_std)
        norm_state_action = tf.concat([norm_state, norm_action], 1)
        norm_delta_state_pred = utils.build_mlp(norm_state_action,
                                                self._state_dim,
                                                "dynamic",
                                                n_layers=self._nn_layers,
                                                reuse=reuse)
        delta_state_pred = utils.unnormalize(norm_delta_state_pred,
                                             self._init_dataset.delta_state_mean,
                                             self._init_dataset.delta_state_std)

        next_state_pred = state + delta_state_pred

        return next_state_pred

    def _reward_func(self, state, action, next_state):
        norm_state = utils.normalize(state,
                                     self._init_dataset.state_mean,
                                     self._init_dataset.state_std)
        norm_action = utils.normalize(action,
                                      self._init_dataset.action_mean,
                                      self._init_dataset.action_std)
        norm_delta_state = utils.normalize(next_state - state,
                                           self._init_dataset.delta_state_mean,
                                           self._init_dataset.delta_state_std)
        norm_state_action = tf.concat([norm_state, norm_action, norm_delta_state], 1)
        norm_reward_pred = utils.build_mlp(norm_state_action,
                                           1,
                                           "reward",
                                           n_layers=self._nn_layers+1,
                                           reuse=tf.AUTO_REUSE)
        reward_pred = utils.unnormalize(norm_reward_pred,
                                        self._init_dataset.reward_mean,
                                        self._init_dataset.reward_std)
        return reward_pred

    def _reward_training(self, reward_ph, reward_pred):

        norm_reward_ph = utils.normalize(reward_ph,
                                         self._init_dataset.reward_mean,
                                         self._init_dataset.reward_std)
        norm_reward_pred = utils.normalize(reward_pred,
                                           self._init_dataset.reward_mean,
                                           self._init_dataset.reward_std)
        loss = tf.reduce_mean(tf.square(norm_reward_pred - norm_reward_ph))
        optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)
        return loss, optimizer

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
            Takes as input the current state, next state, and predicted next state, and returns
            the loss and optimizer for training the dynamics model

            returns:
                loss: Scalar loss tensor
                optimizer: Operation used to perform gradient descent

            implementation details (in order):
                (a) Compute both the actual state difference and the predicted state difference
                (b) Normalize both of these state differences by using the statistics of self._init_dataset and
                    the utils.normalize function
                (c) The loss function is the mean-squared-error between the normalized state difference and
                    normalized predicted state difference
                (d) Create the optimizer by minimizing the loss using the Adam optimizer with self._learning_rate

        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        delta_state_ph = next_state_ph - state_ph
        delta_state_pred = next_state_pred - state_ph
        norm_delta_state_ph = utils.normalize(delta_state_ph,
                                              self._init_dataset.delta_state_mean,
                                              self._init_dataset.delta_state_std)
        norm_delta_state_pred = utils.normalize(delta_state_pred,
                                                self._init_dataset.delta_state_mean,
                                                self._init_dataset.delta_state_std)
        loss = tf.reduce_mean(tf.square(norm_delta_state_pred - norm_delta_state_ph))
        optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

        return loss, optimizer

    def _setup_action_selection(self, state_ph):
        """
            Computes the best action from the current state by using randomly sampled action sequences
            to predict future states, evaluating these predictions according to a cost function,
            selecting the action sequence with the lowest cost, and returning the first action in that sequence

            returns:
                best_action: the action that minimizes the cost function (tensor with shape [self._action_dim])

            implementation details (in order):
                (a) We will assume state_ph has a batch size of 1 whenever action selection is performed
                (b) Randomly sample uniformly self._num_random_action_selection number of action sequences,
                    each of length self._horizon
                (c) Starting from the input state, unroll each action sequence using your neural network
                    dynamics model
                (d) While unrolling the action sequences, keep track of the cost of each action sequence
                    using self._cost_fn
                (e) Find the action sequence with the lowest cost, and return the first action in that sequence

            Hints:
                (i) self._cost_fn takes three arguments: states, actions, and next states. These arguments are
                    2-dimensional tensors, where the 1st dimension is the batch size and the 2nd dimension is the
                    state or action size
                (ii) You should call self._dynamics_func and self._cost_fn a total of self._horizon times
                (iii) Use tf.random_uniform(...) to generate the random action sequences

        """
        ### PROBLEM 2
        ### YOUR CODE HERE
        # raise NotImplementedError
        # current_state = tf.reshape(state_ph, (1, tf.size(state_ph)))

        current_state = tf.tile(state_ph, (self._num_random_action_selection, 1))
        horizon_rewards = 0
        sample_actions = tf.random_uniform(shape=(self._horizon, self._num_random_action_selection, self._action_dim),
                                           minval=self._action_space_low, maxval=self._action_space_high)

        for i in range(self._horizon):
            next_state_pred = self._dynamics_func(current_state, sample_actions[i], reuse=True)
            horizon_rewards = horizon_rewards + self._reward_func(current_state, sample_actions[i], next_state_pred)
            current_state = next_state_pred

        best_action_idx = tf.argmax(horizon_rewards)
        best_action_idx = tf.squeeze(best_action_idx)
        best_action = sample_actions[0][best_action_idx]

        return best_action

    def _setup_graph(self):
        """
        Sets up the tensorflow computation graph for training, prediction, and action selection

        The variables returned will be set as class attributes (see __init__)
        """
        sess = tf.Session()

        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        state_ph, action_ph, next_state_ph, reward_ph = self._setup_placeholders()
        next_state_pred = self._dynamics_func(state_ph, action_ph)
        loss, optimizer = self._setup_training(state_ph, next_state_ph, next_state_pred)

        # fit cost function
        reward_pred = self._reward_func(state_ph, action_ph, next_state_pred)
        reawrd_loss, reward_optimizer = self._reward_training(reward_ph, reward_pred)

        ### PROBLEM 2
        ### YOUR CODE HERE
        # self._rollout_state_ph = tf.placeholder(tf.float32, (1, self._state_dim), name='rollout_state_ph')
        best_action = self._setup_action_selection(state_ph)

        # BONUS
        self._best_action_cross_entropy = self._cross_entropy_action_selection(state_ph)

        sess.run(tf.global_variables_initializer())

        return sess, state_ph, action_ph, next_state_ph, reward_ph, \
            next_state_pred, loss, optimizer, best_action, reward_pred, reawrd_loss, reward_optimizer

    def train_step(self, states, actions, next_states, reward=0):
        """
        Performs one step of gradient descent

        returns:
            loss: the loss from performing gradient descent
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        _, loss, _, _ = self._sess.run([self._optimizer, self._loss, self._reward_optimizer, self._reward_loss],
                                       feed_dict={self._state_ph: np.atleast_2d(states),
                                                  self._action_ph: np.atleast_2d(actions),
                                                  self._next_state_ph: np.atleast_2d(next_states),
                                                  self._reward_ph: np.atleast_2d(reward)})
        return loss

    def predict(self, state, action):
        """
        Predicts the next state given the current state and action

        returns:
            next_state_pred: predicted next state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)

        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        next_state_pred = self._sess.run(self._next_state_pred,
                                         feed_dict={self._state_ph: np.atleast_2d(state),
                                                    self._action_ph: np.atleast_2d(action)})
        next_state_pred = next_state_pred[0]

        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def predict_reward(self, state, action, next_state):
        """
        Predicts the reward given state, action and next_state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)
        assert np.shape(next_state) == (self._state_dim,)

        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        reward_pred = self._sess.run(self._reward_pred,
                                     feed_dict={self._state_ph: np.atleast_2d(state),
                                                self._action_ph: np.atleast_2d(action),
                                                self._next_state_ph: np.atleast_2d(action)})
        reward_pred = reward_pred[0]
        assert np.shape(reward_pred) == (self._state_dim,)
        return reward_pred

    def get_action(self, state):
        """
        Computes the action that minimizes the cost function given the current state

        returns:
            best_action: the best action
        """
        assert np.shape(state) == (self._state_dim,)

        ### PROBLEM 2
        ### YOUR CODE HERE
        # raise NotImplementedError
        best_action = self._sess.run(self._best_action,
                                     feed_dict={self._state_ph: np.atleast_2d(state)})
        assert np.shape(best_action) == (self._action_dim,)

        return best_action

    def get_action_cross_entropy(self, state):
        assert np.shape(state) == (self._state_dim,)

        ### PROBLEM 2
        ### YOUR CODE HERE
        # raise NotImplementedError
        best_action = self._sess.run(self._best_action_cross_entropy,
                                     feed_dict={self._state_ph: np.atleast_2d(state)})
        assert np.shape(best_action) == (self._action_dim,)

        return best_action

    def _cross_entropy_action_selection(self, state_ph):

        mu = (self._action_space_low + self._action_space_high) / 2
        sigma2 = (self._action_space_high - self._action_space_low) ** 2 / 12

        N = 200
        Ne = 20
        state_ph_tiled = tf.tile(state_ph, (N, 1))
        t = 0

        all_horizon_rewards = []
        all_first_actions = []

        while t < self._num_random_action_selection:

            if t + N > self._num_random_action_selection:
                N = self._num_random_action_selection - t
                if Ne > N:
                    Ne = N
                current_state = tf.tile(state_ph, (N, 1))
            else:
                current_state = state_ph_tiled

            horizon_rewards = 0
            sample_actions = tf.clip_by_value(tf.random_normal(shape=(self._horizon, N, self._action_dim),
                                                               mean=mu,
                                                               stddev=sigma2),
                                              self._action_space_low,
                                              self._action_space_high)
            all_first_actions.append(sample_actions[0])
            for i in range(self._horizon):
                next_state_pred = self._dynamics_func(current_state, sample_actions[i], reuse=True)
                horizon_rewards += self._reward_func(current_state, sample_actions[i], next_state_pred)
                current_state = next_state_pred

            all_horizon_rewards.append(horizon_rewards)
            idx = tf.contrib.framework.argsort(horizon_rewards)
            idx = idx[1:Ne]
            X = tf.gather(sample_actions, idx, axis=1)
            X = tf.reshape(X, shape=(-1, self._action_dim))
            mu, sigma2 = tf.nn.moments(X, axes=[0])
            sigma2 = tf.clip_by_value(sigma2, 1e-8, 1e2)

            t += N

        best_action_idx = tf.argmin(tf.concat(all_horizon_rewards, 0))
        best_action_idx = tf.squeeze(best_action_idx)
        all_first_actions = tf.concat(all_first_actions, 0)
        best_action = all_first_actions[best_action_idx]

        return best_action
