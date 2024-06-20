"""
Definition of the Global Average Pooling Layer for the Particle Cloud and Point Net NN.
It takes the feature-wise average over all particles in the event.
"""

import tensorflow as tf
import keras


class GlobalAveragePoolingLayer(keras.layers.Layer):
    """Takes the feature-wise average over all particles in the event"""

    def call(self, events):
        """
        Calculates the feature-wise average over all the real particles in the event.
        Zero-padded particles are not included. We assume the mask is the last particle feature.
        """
        # The last feature of each real particle must be one
        real_particles_mask = tf.equal(events[:, :, -1], 1)
        # Creating a tensor with only the real particles
        real_particles = tf.ragged.boolean_mask(events, real_particles_mask)
        # Calculates the average over all particles and do not return the mask
        return tf.reduce_mean(real_particles, axis=1)[:, :-1]

    def compute_output_shape(self, input_shape):
        """
        Returns the output shape of the layer.
        It's equal to the number of events (batch size) and number of particles features - 1 (excluding the mask)
        """
        batch_size, particles_features = input_shape[0], input_shape[-1]
        return batch_size, particles_features - 1
