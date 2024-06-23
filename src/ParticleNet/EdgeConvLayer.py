"""
Definition of the Edge Convolution Layer that it's used in the Particle Net (Particle Cloud NN).

The ayer first finds the k-nearest neighbors of a given particle. Each pair of particle and
a neighbor is call an edge. The Edge Convolution Layer (EdgeConvLayer) evaluates a Multi-Layer Perceptron
on each edge and takes the average over of all the outputs of the MLP over all the edges.
The average over each output defines the new set of features for the particle.
"""

import keras
import tensorflow as tf


class EdgeConvLayer(keras.layers.Layer):
    """
    Creates a EdgeConv layer with a single MLP.
    An EdgeConv layer with more than one MLP can be created stacking different
    EdgeConvolution layers objects.
    """

    def __init__(self, mlp, n_particles_features, mlp_output_dim, final_index_coord, max_number_particles, k_neighbors,
                 **kwargs):
        super().__init__(**kwargs)
        # stores the MLP that will be used as the edge function
        self._mlp = mlp
        self._mlp_output_dim = mlp_output_dim
        self._k_neighbors = k_neighbors
        self._final_index_coord = final_index_coord
        self._max_number_particles = max_number_particles
        # number of particles features
        self._n_particles_features = n_particles_features
        # relu activation function
        self._leaky_relu = keras.activations.leaky_relu

    def call(self, events):
        """Evaluates the EdgeConvolution layer on each event sample."""
        return tf.map_fn(self._edge_conv_layer, events, dtype=tf.float32)

    def _edge_conv_layer(self, particles):
        """Evaluates the EdgeConvolution layer an event sample."""
        # only non zero padded particles
        real_particles = self._get_real_particles(particles)
        # number of real particles in the event
        number_of_particles = tf.shape(real_particles)[0]

        # getting the coordinates of each particle and using them to find the neighbors
        coordinates = real_particles[:, : self._final_index_coord]
        neighbors_indices = self._find_neighbors(coordinates)

        # creating the features for each edge
        # excluding the last column since we do not need anymore
        edges = self._create_edge_features(real_particles[:, :-1], neighbors_indices)

        # evaluating the MLP on each edge
        mlp_output = self._mlp(edges)
        # taking the feature-wise average over all the edges for a particle
        final_cloud_particles = self._feature_wise_average(mlp_output, number_of_particles)

        # aggregation step: new features together with the old features
        new_particles_features = self._aggregate_features(final_cloud_particles, real_particles[:, :-1])

        # using the final activation function
        new_particles_features = self._leaky_relu(new_particles_features)

        # we need to ensure that the size of the output is fixed
        # to do so, we add a mask column (1 True particles, 0 ZeroPadded particles)
        return self._fix_number_of_particles(new_particles_features)

    def _find_neighbors(self, coordinates):
        """Finds the k-nearest neighbors for all the particles"""
        # calculating the distance among all the particles
        # (this creates a tensor with (n_particles, n_particles, coords)
        diffence_from_central_particle = tf.expand_dims(coordinates, axis=1) - tf.expand_dims(coordinates, axis=0)
        distances = tf.norm(diffence_from_central_particle, axis=-1)
        # index of the own particle + the closest k-neighbors
        # top_k returns the k highest indices, using -distance makes sure that we are returning the closest
        # particles
        _, neigh_indices = tf.math.top_k(-distances, self._k_neighbors + 1)
        # returning only the k-closest neighbors for each particle and exluding the own particle
        return neigh_indices[:, 1:]

    @staticmethod
    def _get_real_particles(particles):
        """Removes zero padded mask == 0 - assumes the last column is the mask"""
        real_particles = tf.equal(particles[:, -1], 1)
        # returns only the real particles
        return tf.boolean_mask(particles, real_particles)

    def _create_edge_features(self, particles, neighbor_indices):
        """Creates the feature vector for each edge (a particle and one neighbor)."""
        # since for each edge we must have the information about the central particle
        # we define the central particle tensor as a tensor with each central particle repeated k times
        central_particles = tf.repeat(particles, self._k_neighbors, axis=0)

        # getting the neighbors
        neighbor_indices_flat = tf.reshape(neighbor_indices, shape=(-1,))
        neighbors_features = tf.gather(particles, neighbor_indices_flat)

        # calculating the difference from the central particle
        diff_from_particle = neighbors_features - central_particles

        # the edge feature is the vector [central particle features, relative features]
        return tf.concat([central_particles, diff_from_particle], axis=1)

    def _feature_wise_average(self, edges, number_of_particles):
        """Evaluates the feature-wise average for every particle cloud"""
        # reshaping the edges to separate every particle cloud
        particle_cloud = tf.reshape(edges, shape=(number_of_particles, self._k_neighbors, self._mlp_output_dim))
        # taking the average over every cloud
        return tf.reduce_mean(particle_cloud, axis=1)

    def _fix_number_of_particles(self, particles):
        """
        Checks if the total number of particles is equal to max_number_particles.
        If it's not, if adds vectors with zero entries to match the size of the matrix
        """
        number_of_particles = tf.shape(particles)[0]
        # first we need to create a mask and set it to 1 for all true particles
        mask_true_particles = tf.ones(shape=[number_of_particles, 1], dtype=tf.float32)
        masked_particles = tf.concat([particles, mask_true_particles], axis=1)
        # adjusting the vector
        number_of_particles_to_add = self._max_number_particles - number_of_particles
        zero_padded_particles = tf.zeros([number_of_particles_to_add, tf.shape(masked_particles)[1]], dtype=tf.float32)

        return tf.concat([masked_particles, zero_padded_particles], axis=0)

    @staticmethod
    def _aggregate_features(particle_cloud, particles):
        """Aggregates the features of the particle cloud with the particle features."""
        return tf.concat([particle_cloud, particles], axis=-1)

    def compute_output_shape(self, input_shape):
        """Returns the shape of the output tensor."""
        batch_size = input_shape[0]
        # the last plus one is because we are adding the mask
        return batch_size, self._max_number_particles, self._mlp_output_dim + self._n_particles_features + 1

    def get_config(self):
        """Configurations of the NN besides the default ones"""
        # base configurations
        base_config = super().get_config()
        # custom configurations
        config = {
            "mlp": keras.saving.serialize_keras_object(self._mlp),
            "mlp_output_dim": self._mlp_output_dim,
            "final_index_coord": self._final_index_coord,
            "max_number_particles": self._max_number_particles,
            "k_neighbors": self._k_neighbors,
            "n_particles_features": self._n_particles_features
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Extraciting the MLP when loading the model."""
        mlp = config.pop("mlp")
        mlp = keras.saving.deserialize_keras_object(mlp)
        return cls(mlp, **config)
