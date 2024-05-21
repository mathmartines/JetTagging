from src.Particle import ParticleType, Jet, JetConstituent
from vector.backends.object import MomentumObject3D
import pandas as pd


class JetBuilder:
    """
    This class reads a file that contains all the information concerning a jet originated from a
    single particle (Top, gluon, etc.). And it creates a list of Jet objects with its constituents
    given in the file.
    """

    def __init__(self, data_frame: pd.DataFrame, particle_type: ParticleType):
        self._data = data_frame
        self._jet_type = particle_type
        self._jet_constituent_type = {1: ParticleType.Parton, 0: ParticleType.ZeroPadded}

    def create_jets(self):
        """Creates a numpy array to store all the jets in the file."""
        return [self._create_jet(jet_constituents) for _, jet_constituents in self._data.iterrows()]

    def _create_jet(self, jet_constituents: pd.Series) -> Jet:
        """Creates a Jet object"""
        # number of constituents is the size of the series divided by the number of features (4)
        number_of_jets = int(len(jet_constituents) / 4)
        jet = Jet(particle_type=self._jet_type, n_constituents=number_of_jets)

        # Creating the JetConstituents objects and adding them to the jet
        jet_constituents_array = jet_constituents.to_numpy()
        for index_constituent in range(0, number_of_jets):
            # Each particle has four feature, and all the particles for a given jet are in a single row
            # This means that the first particle has feature from 0 to 3, the second from 4 to 7, and so on.
            eta, phi, pt, mask = jet_constituents_array[4 * index_constituent: 4 * index_constituent + 4]
            jet_constituent = JetConstituent(
                particle_type=self._jet_constituent_type[mask],
                momentum=MomentumObject3D(pt=pt, eta=eta, phi=phi)
            )
            jet.add_particle(constituent=jet_constituent)

        return jet

