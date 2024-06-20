"""
Definition of the convolution layer for PointNet architecture.

The layer basically sees the event as a set of particles, and evaluates
a Multi-Layer Perceptron (MLP) in each particle in the event.

Basically, this layer is responsible to create new features for the
particle. Also, it can be stacked, since the output of the layer
can be seen as the same set of particles, but with new features
defined by the output of the MLP.
"""

import tensorflow as tf
import keras


class PointNetLayer(keras.layers.Layer):
    """Definition of the PointNet layer for the PointNet architecture."""

    def __init__(self, mlp, output_dim, **kwargs):
        super().__init__(**kwargs)
        # MLP that will be used to evaluate each particle
        self._mlp = mlp
        # Number of output features of the MLP
        self._mlp_output_dim = output_dim

    def call(self, events):
        """
        Evaluates the MLP on all particles in all the events.
        We assume that the last particle feature is a mask that tells whether the particle is real
        or zero-padded. So the last column will not enter as an input to the MLP.

        :param events: Tensor of shape (batch_size, num_particles, num_features)
                       representing a list of events (sets of particles).
        :return: Tensor of shape (batch_size, num_particles, mlp_output_dim + 1)
                 where each particle has a new set of features defined by the MLP.
        """
        # shape of the input events
        input_shape = tf.shape(events)
        num_of_events, num_particles_per_event, num_particles_features = input_shape[0], input_shape[1], input_shape[2]
        # Reshape to have a list of particles
        all_particles = tf.reshape(events, (-1, num_particles_features))
        # Find real particles and their indices
        real_particles, real_particles_indices = self._get_real_particles(all_particles)
        # Evaluates the MLP on all the real particles
        output_mlp = self._apply_mlp(real_particles)
        # Create the output tensor with all particles
        output_particles = self._generate_output_particles(tf.shape(all_particles)[0], output_mlp,
                                                           real_particles_indices)
        # Reshape to (number_of_evts, number_of_particles_per_evt, mlp_output_dim + 1)
        return tf.reshape(output_particles, (num_of_events, num_particles_per_event, self._mlp_output_dim + 1))

    @staticmethod
    def _get_real_particles(particles):
        """Finds the real particles and their respective indices."""
        real_particle_cond = tf.equal(particles[:, -1], 1)
        real_particles = tf.boolean_mask(particles, real_particle_cond)
        real_particle_indices = tf.where(real_particle_cond)
        return real_particles, real_particle_indices

    def _apply_mlp(self, particles):
        """Applies the MLP over all the particles"""
        # Evaluate the MLP the particles
        output_mlp = self._mlp(particles[:, :-1])  # Exclude the mask feature for MLP input
        mask = tf.ones((tf.shape(output_mlp)[0], 1), dtype=tf.float32)
        # return the new particles features + the mask
        return tf.concat([output_mlp, mask], axis=1)

    def _generate_output_particles(self, total_number_of_particles, output_mlp, real_particle_indices):
        """Generate the particles outputs."""
        output_particles = tf.zeros((total_number_of_particles, self._mlp_output_dim + 1), dtype=tf.float32)
        return tf.tensor_scatter_nd_update(output_particles, real_particle_indices, output_mlp)

    def compute_output_shape(self, input_shape):
        """For each sample the output shape is (num_particles, mlp_output_dim + 1)"""
        batch_size = input_shape[0]
        number_of_particles = input_shape[1]
        return batch_size, number_of_particles, self._mlp_output_dim + 1

    def get_config(self):
        """Configurations of the NN besides the default ones"""
        # base configurations
        base_config = super().get_config()
        # custom configurations
        config = {
            "mlp": keras.saving.serialize_keras_object(self._mlp),
            "mlp_output_dim": self._mlp_output_dim
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Extraciting the MLP when loading the model."""
        mlp = config.pop("mlp")
        mlp = keras.saving.deserialize_keras_object(mlp)
        return cls(mlp, **config)
