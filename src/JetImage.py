from __future__ import annotations
from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod
from src.Particle import Jet


class JetImage:
    """
    Class responsible to set up the phi x eta grid, where  each bin will
    store the jet pt intensity in that range. That is, each phi x eta pixel
    will store the pt sum of all the jet's constituents that have phi and eta
    values inside the bin region.

    It must receive the range of each variable: (phi_min, phi_max), (eta_min, eta_max);
    and it also needs the information about the bin size in the phi and eta axis.

    The main method to set up the grid is the create_jet_image, where it receives a
    jet as input and returns the grid containing the jet pt intensity in each bin.
    """

    def __init__(self,
                 phi_range: Tuple[float, float],
                 eta_range: Tuple[float, float],
                 n_bins_phi: int,
                 n_bins_eta: int,
                 pt_intensity_calculator: IntensityPtCalculator
                 ):
        self._phi_min, self._phi_max = phi_range
        self._eta_min, self._eta_max = eta_range
        self._n_bins_phi = n_bins_phi
        self._n_bins_eta = n_bins_eta
        # creating the grid
        # x - eta number of bins, y - phi values
        self._jet_image = np.zeros(shape=self._n_bins_eta * self._n_bins_phi)
        # evaluating the bin sizes
        self._eta_bin_size = (self._eta_max - self._eta_min) / self._n_bins_eta
        self._phi_bin_size = (self._phi_max - self._phi_min) / self._n_bins_phi
        # tells how the pt intesity in an angle region should be computed
        self._pt_intensity_calculator = pt_intensity_calculator
        self._pt_intensity_calculator.set_jet_image(self)

    @property
    def jet_image(self):
        return self._jet_image

    def update_jet_image(self, eta_value: float, phi_value: float, pt_value: float):
        """Adds the pT value to the bin containing the phi and eta values."""
        # the eta bin starts from the maximum value and goes to the minimum
        eta_bin = self._n_bins_eta - int((eta_value - self._eta_min) / self._eta_bin_size) - 1
        # the phi bin starts from the minimum value
        phi_bin = int((phi_value - self._phi_min) / self._phi_bin_size)
        # updating the jet image only if it's in the allowed range
        if 0 <= eta_bin < self._n_bins_eta and 0 <= phi_bin < self._n_bins_phi:
            self._jet_image[eta_bin * self._n_bins_phi + phi_bin] += pt_value

    def create_jet_image(self, jet) -> np.ndarray:
        """Creates the jet pt image"""
        # reseting the grid (in case it had been used before)
        self._jet_image[:] = 0
        # perfoming the calculation of the intensity in each bin
        self._pt_intensity_calculator.calculate_intensity(jet)
        # reshaping the array in a convinient way to the user
        return np.reshape(self._jet_image, newshape=(self._n_bins_eta, self._n_bins_phi))


class IntensityPtCalculator(ABC):
    """
    Abstract class which defines the strategy that we want to evaluate the jet pt intensity in a fixed angle
    region.
    """

    def __init__(self):
        self._jet_image = None

    @abstractmethod
    def calculate_intensity(self, jet):
        """Evaluates the jet pT intensity in the fixed angle region."""
        raise NotImplementedError

    def set_jet_image(self, jet_image: JetImage):
        self._jet_image = jet_image


class JetImageCalculator(IntensityPtCalculator):
    """Evaluates the pT intensity in each pixel of a Jet object"""
    def calculate_intensity(self, jet: Jet):
        """Updates the jet image with the jet constituents pT"""
        for jet_constituent in jet:
            jet_momentum = jet_constituent.momentum
            self._jet_image.update_jet_image(
                eta_value=jet_momentum.eta, phi_value=jet_momentum.phi, pt_value=jet_momentum.pt
            )
