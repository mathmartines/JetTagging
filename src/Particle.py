from abc import ABC, abstractmethod
from vector.backends.object import MomentumObject3D
from enum import Enum
import numpy as np


class ParticleType(Enum):
    ZeroPadded = 0
    Parton = 1
    Top = 2
    LightQuark = 3
    Gluon = 4


class Particle(ABC):
    """
    Abstract class to represent a particle.

    A particle must have a momentum and a ParticleType. The momentum is given by the
    vector.MomentumObject3D object.

    This abstraction can represent the Jet consituents and the Jet itself.
    """

    def __init__(self, particle_type: ParticleType):
        self._particle_type = particle_type

    @property
    @abstractmethod
    def momentum(self) -> MomentumObject3D:
        """Returns the momentum of the particle"""
        pass

    @property
    def particle_type(self) -> ParticleType:
        return self._particle_type


class JetConstituent(Particle):
    """
    Class to represent a Jet constituent.

    It must have a 3D-vector representing the momentum and a ParticleType. The latter can be Parton or ZeroPadded.
    When the ParticleType is ZeroPadded, it means that it does not represent a real particle.
    """

    def __init__(self, particle_type: ParticleType, momentum: MomentumObject3D = MomentumObject3D(pt=0, eta=0, phi=0)):
        self._momentum = momentum
        super().__init__(self._validate_particle_type(particle_type))

    @property
    def momentum(self) -> MomentumObject3D:
        return self._momentum

    @staticmethod
    def _validate_particle_type(particle_type: ParticleType) -> ParticleType:
        """Returns True if the particle type is a valid one."""
        if particle_type in [ParticleType.ZeroPadded, ParticleType.Parton]:
            return particle_type
        print(f"{particle_type} is not a valid type for this class")
        return ParticleType.ZeroPadded

    def __repr__(self):
        return f"JetConstituent({self.particle_type}, {self._momentum})"


class Jet(Particle):
    """
    Class to represent a Jet.

    This class stores all the particles that makes the jet.
    The Jet also have a particle type, which represents the particle that the Jet originated from.

    The momentum of the jet is the sum of the momentum of all its constituents.
    """

    def __init__(self, particle_type: ParticleType, n_constituents: int):
        self._constituents = np.empty(shape=n_constituents, dtype=Particle)
        self._index_pos = 0
        self._momentum = MomentumObject3D(pt=0, eta=0, phi=0)
        super().__init__(particle_type)

    def add_particle(self, constituent: Particle):
        """Add a particle to the list of particle constituents."""
        # checking if it's a real particle
        # in case we need more space we need to resize the vector
        self._check_vector_lenght()
        self._constituents[self._index_pos] = constituent
        self._index_pos += 1
        # adding the particle momentum to the total momentum of the jet
        self._update_momentum(constituent.momentum)

    def _update_momentum(self, momentum_const: MomentumObject3D):
        self._momentum.pt += momentum_const.pt
        self._momentum.phi += momentum_const.phi
        self._momentum.eta += momentum_const.eta

    def jet_substructure(self) -> np.array:
        return np.array([
            [particle.momentum.pt, particle.momentum.eta, particle.momentum.phi]
            for particle in self._constituents if particle.particle_type != ParticleType.ZeroPadded
        ])

    def _check_vector_lenght(self):
        """Checks if we need to add more space to the vector."""
        if self._index_pos == len(self._constituents):
            self._constituents = np.resize(self._constituents, len(self._constituents) + 1)

    def __repr__(self):
        return f"Jet({self.particle_type}, {self._momentum}, n_constituents: {len(self)})"

    @property
    def momentum(self) -> MomentumObject3D:
        return self._momentum

    def __len__(self):
        """Returns the number of non-ZeroPadded particles."""
        return len([particle for particle in self._constituents if particle.particle_type != ParticleType.ZeroPadded])

    def __iter__(self):
        return iter(self._constituents)

    def __getitem__(self, index: int):
        return self._constituents[index]
