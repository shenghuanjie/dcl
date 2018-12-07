import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python import keras
from tensorflow.python.keras.engine.network import Network
from tensorflow_probability import distributions
from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow.keras.backend as K


class QFunction(Network):
    def __init__(self, hidden_layer_sizes, **kwargs):
        super(QFunction, self).__init__(**kwargs)
        self._hidden_layer_sizes = hidden_layer_sizes

    def build(self, input_shape):
        inputs = [
            layers.Input(batch_shape=input_shape[0], name='observations'),
            layers.Input(batch_shape=input_shape[1], name='actions')
        ]

        x = layers.Concatenate(axis=1)(inputs)
        for hidden_units in self._hidden_layer_sizes:
            x = layers.Dense(hidden_units, activation='relu')(x)
        q_values = layers.Dense(1, activation=None)(x)

        self._init_graph_network(inputs, q_values)
        super(QFunction, self).build(input_shape)


class ValueFunction(Network):
    def __init__(self, hidden_layer_sizes, **kwargs):
        super(ValueFunction, self).__init__(**kwargs)
        self._hidden_layer_sizes = hidden_layer_sizes

    def build(self, input_shape):
        inputs = layers.Input(batch_shape=input_shape, name='observations')

        x = inputs
        for hidden_units in self._hidden_layer_sizes:
            x = layers.Dense(hidden_units, activation='relu')(x)
        values = layers.Dense(1, activation=None)(x)

        self._init_graph_network(inputs, values)
        super(ValueFunction, self).build(input_shape)


class DistributionLayer(Layer):

    def __init__(self, reparameterize=True, **kwargs):
        super().__init__(**kwargs)
        self._reparameterize = reparameterize

    def create_distribution_layer(self, mean_and_log_std):
        mean, log_std = tf.split(
            mean_and_log_std, num_or_size_splits=2, axis=1)
        log_std = tf.clip_by_value(log_std, -20., 2.)

        distribution = distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(log_std))

        raw_actions = distribution.sample()
        if not self._reparameterize:
            ### Problem 1.3.A
            ### YOUR CODE HERE
            # raise NotImplementedError
            raw_actions = tf.stop_gradient(raw_actions)
        log_probs = distribution.log_prob(raw_actions)
        log_probs -= self._squash_correction(raw_actions)

        # actions = None
        ### Problem 2.A
        ### YOUR CODE HERE
        # raise NotImplementedError
        actions = tf.tanh(raw_actions)

        return actions, log_probs

    def _squash_correction(self, raw_actions, eps=1e-8, stable=True):
        ### Problem 2.B
        ### YOUR CODE HERE
        # raise NotImplementedError
        if stable:
            return tf.reduce_sum(tf.log(4.0) -
                                 2.0 * (tf.nn.softplus(2.0 * raw_actions) - raw_actions),
                                 axis=1)
        else:
            return tf.reduce_sum(tf.log(1 - tf.square(tf.tanh(raw_actions)) + eps), axis=1)

    def call(self, *args):
        # calculations with inputTensor and the weights you defined in "build"
        # inputTensor may be a single tensor or a list of tensors

        # output can also be a single tensor or a list of tensors
        return self.create_distribution_layer(*args)


class GaussianPolicy(Network):
    def __init__(self, action_dim, hidden_layer_sizes, reparameterize, **kwargs):
        super(GaussianPolicy, self).__init__(**kwargs)
        self._action_dim = action_dim
        self._f = None
        self._hidden_layer_sizes = hidden_layer_sizes
        self._reparameterize = reparameterize

    def build(self, input_shape):
        inputs = layers.Input(batch_shape=input_shape, name='observations')

        x = inputs
        for hidden_units in self._hidden_layer_sizes:
            x = layers.Dense(hidden_units, activation='relu')(x)

        mean_and_log_std = layers.Dense(
            self._action_dim * 2, activation=None)(x)

        samples, log_probs = DistributionLayer(reparameterize=self._reparameterize)\
            (mean_and_log_std)

        self._init_graph_network(inputs=inputs, outputs=[samples, log_probs])
        super(GaussianPolicy, self).build(input_shape)

    def eval(self, observation):
        assert self.built and observation.ndim == 1

        if self._f is None:
            self._f = keras.backend.function(self.inputs, [self.outputs[0]])

        action, = self._f([observation[None]])
        return action.flatten()

    # def get_config(self):
    #     config = super().get_config()
    #     config['distribution_layer'] = self.create_distribution_layer # say self. _localization_net  if you store the argument in __init__
    #     return config