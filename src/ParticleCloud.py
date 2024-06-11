from tensorflow.keras import Layer
import tensorflow as tf


class EdgeConvolutionLayer(Layer):
    """
    Creates a EdgeConv layer with a single MLP.
    An EdgeConv layer with more than one MLP can be created stacking different
    EdgeConvolution layers objects.
    """

    def __init__(self, mlp: Layer, final_index_coord, max_number_particles, k_neighbors, **kwargs):
        super().__init__(**kwargs)
        # stores the Multi-Layer perceptron that will be used for each particle cloud
        self._mlp = mlp
        self._mlp_output_dim = mlp.output_shape[-1]
        self._k_neighbors = k_neighbors
        self._final_index_coord = final_index_coord
        self._max_number_particles = max_number_particles

    def call(self, inputs):
        outputs = tf.map_fn(self.call_on_sample, inputs, dtype=tf.float32)
        return outputs

    def call_on_sample(self, jet_set):
        # only non zero padded particles
        jet_set = self.remove_zero_padded_particles(jet_set)

        coordinates, features = jet_set[:, 0:self._final_index_coord], jet_set[:, self._final_index_coord:]
        number_of_particles = tf.shape(coordinates)[0]
        # calculating the indices of the neighbors for each particle
        neighbors_indices = self._find_neighbors(coordinates)

        # for each particle in the set, we need to create k_neighbors connections
        # that is, we need to create n_particles * k_neighbors input data to the MLP
        particles_coord = tf.repeat(coordinates, repeats=self._k_neighbors, axis=0)
        particles_features = tf.repeat(features, repeats=self._k_neighbors, axis=0)

        # now we can get each particle's neighbor by using the gather method
        neighbors_indices_flat = tf.reshape(neighbors_indices, [-1])
        neighbors_coord = tf.gather(coordinates, neighbors_indices_flat)
        neighbors_features = tf.gather(features, neighbors_indices_flat)
        neighbors_coord = tf.reshape(neighbors_coord, [number_of_particles * self._k_neighbors, -1])
        neighbors_features = tf.reshape(neighbors_features, [number_of_particles * self._k_neighbors, -1])

        # setting up the features for the shared MLP
        edges_features = self._create_feature_for_edge(
            particles_coord, neighbors_coord, particles_features, neighbors_features
        )
        # we need to reshape the result of the MLP
        mlp_output = self._mlp(edges_features)
        mlp_output_per_particle = tf.reshape(mlp_output, [number_of_particles, self._k_neighbors, self._mlp_output_dim])

        # caculating the avarage for each particle
        avg_per_particle = tf.reduce_mean(mlp_output_per_particle, axis=1)

        # we need to ensure that the size of the output is fixed
        # to do so, we add a mask column (1 True particles, 0 ZeroPadded particles)
        avg_per_particle = self._fix_number_of_particles(avg_per_particle)

        # we follow the Jet Tagging via Particle Clouds in each they take the global avarage
        return avg_per_particle

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
    def _create_feature_for_edge(particles_coordinates, neighbors_coordinates, particles_features, neighbors_features):
        """Creates the feature vector for the pair particle, neighbor"""
        neighbors_features_diff = neighbors_features - particles_features
        neighbors_coordinates_diff = neighbors_coordinates - particles_coordinates
        # note that here we are following the idea (x_i, x_ij - x_i)
        # TODO: thing if this is indeed the best way
        return tf.concat(
            [particles_coordinates, particles_features, neighbors_coordinates_diff, neighbors_features_diff],
            axis=-1
        )

    @staticmethod
    def remove_zero_padded_particles(input_particles):
        """Removes zero padded mask == 0 - assumes the last column is the mask"""
        zero_padded_particles = tf.where(input_particles[:, -1] == 0)
        has_zero_padded_particles = tf.shape(zero_padded_particles)[0] > 0

        def remove_zeros():
            return input_particles[:zero_padded_particles[0, 0], :-1]

        def keep_all():
            return input_particles[:, :-1]

        return tf.cond(has_zero_padded_particles, input_particles[:zero_padded_particles[0, 0], :-1], keep_all)

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
        zero_padded_particles = tf.zeros([number_of_particles_to_add, masked_particles.shape[1]], dtype=tf.float32)

        return tf.concat([masked_particles, zero_padded_particles], axis=0)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        # the last plus one is because we are adding the mask
        return batch_size, self._max_number_particles, self._mlp_output_dim + 1


class ChannelWiseGlobalAvaragePooling(tf.keras.layers.Layer):

    def call(self, inputs):
        return tf.map_fn(self.call_on_sample, inputs, dtype=tf.float32)

    @staticmethod
    def call_on_sample(input_sample):
        # removing the zero padded particles (we can use the static method from the previous class)
        particles = EdgeConvolutionLayer.remove_zero_padded_particles(input_sample)
        # calculating the channel-wise avarage for each particle
        return tf.reduce_mean(particles, axis=0)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        # the output shape is the number of features of each particle - 1 (not including the mask)
        return batch_size, input_shape[-1] - 1

# class EdgeConvolutionLayerSimple(Layer):
#     """Implements the Edge Convolution Layer in a simple way - not optimized"""
#
#     def __init__(self, mlp: Layer, final_index_coord, k_neighbors, **kwargs):
#         super().__init__(**kwargs)
#         # stores the Multi-Layer perceptron that will be used for each particle cloud
#         self._mlp = mlp
#         self._k_neighbors = k_neighbors
#         self._final_index_coord = final_index_coord
#
#     def call(self, inputs):
#         # outputs = tf.map_fn(self.call_on_sample, inputs,  fn_output_signature=tf.float32)
#         # outputs = np.array([self.call_on_sample(sample) for sample in inputs])
#         return tf.map_fn(self.call_on_sample, inputs, dtype=tf.float32)
#
#     @tf.function
#     def call_on_sample(self, sample_input):
#         """Calculates the edge convolution layer in a single input"""
#         # coordinates, features = sample_input[:, :self._k_neighbors], sample_input[:, self._k_neighbors:]
#         coordinates = sample_input[:, :self._final_index_coord]
#         # the output space is going to be a matrix with shape (n_particles, output dim from MLP)
#         # particle_cloud = tf.Variable(tf.zeros([coordinates.shape[0], self._mlp.output_dim[-1]]), dtype=tf.float32)
#         # particle_cloud = tf.TensorArray(tf.float32, size=coordinates.shape[0])
#         # particle_cloud = np.empty((sample_input.shape[0], self._mlp.output_shape[-1]), dtype=np.float32)
#         particle_cloud = tf.TensorArray(tf.float32, size=sample_input.shape[0])
#
#         # finding the k-neighbors for each particle
#         for particle_index in range(coordinates.shape[0]):
#             # cloud_output = tf.Variable(tf.zeros([self._k_neighbors, self._mlp.output_dim[-1]]), dtype=tf.float32)
#             cloud_output = tf.TensorArray(tf.float32, size=self._k_neighbors)
#             # cloud_output = np.empty((self._k_neighbors, self._mlp.output_shape[-1]))
#             for neighbor_index, index_cloud in zip(self._find_neighbors(particle_index, coordinates),
#                                                    range(self._k_neighbors)):
#                 # setting up the features for each edge (eah pair of particles)
#                 # edge_features = self._create_edge_features(sample_input[particle_index], sample_input[neighbor_index])
#                 edge_features = self._create_edge_features(sample_input[particle_index], sample_input[neighbor_index])
#                 cloud_output = cloud_output.write(index_cloud, self._mlp(edge_features)[0])
#                 # cloud_output[index_cloud] = self._mlp(edge_features)[0]
#                 # cloud_output[index_cloud].assign(self._mlp(edge_features))
#             # perfoming the channel-wise symmetric computation
#             # particle_cloud[particle_index].assign(tf.reduce_mean(cloud_output, axis=0))
#             particle_cloud = particle_cloud.write(particle_index, tf.reduce_mean(cloud_output.stack(), axis=0))
#             # particle_cloud[particle_index] = np.mean(cloud_output, axis=0)
#         # print(particle_cloud)
#         # we finally return the particle cloud with the new features
#         return particle_cloud.stack()
#
#     @staticmethod
#     def _create_edge_features(particle_input, neighbor_input):
#         """Set up the features for the MLP evaluation"""
#         # here we use the same definition as is in the paper
#         return tf.reshape(tf.concat([particle_input, neighbor_input - particle_input], axis=0), [1, 6])
#
#     def _find_neighbors(self, particle_index, coordinates):
#         """Finds the k-nearest neighbors of particle at index particle_index"""
#         # distance_from_particle = np.linalg.norm(coordinates - coordinates[particle_index], axis=1)
#         # # fiding the k-nearest particle (not including the own particle
#         # return np.argsort(distance_from_particle)[1:self._k_neighbors + 1]
#         distance_from_particle = tf.norm(coordinates - coordinates[particle_index], axis=1)
#         # Use tf.math.top_k instead of np.argsort
#         _, indices = tf.math.top_k(distance_from_particle, k=self._k_neighbors + 1)
#         return indices[1:]
#
#     # def compute_output_shape(self, input_shape):
#     #     batch_size = input_shape[0]
#     #     output_dim = (input_shape[1], self._mlp.output_shape[-1])
#     #     return batch_size, output_dim
