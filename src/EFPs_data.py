"""
Running the EFPs on the full dataset takes a long time to run it all once and store them in a input data, instead
of performing the calculation many times.
"""

from src.Preprocessing import PreprocessingEFPs, create_labels_single_column
from src.Particle import ParticleType
import pandas as pd
import numpy as np


if __name__ == "__main__":
    # reading the data we want to save
    data_file = pd.read_csv('../../Data/g_jets.csv', header=None, sep=' ')
    # defining the degree of the polynomions
    efp_processing = PreprocessingEFPs(5, create_labels_single_column)
    # processing the data by constructing the polynomials
    # this takes a while to run
    X = efp_processing.transform(X=data_file.to_numpy(), y={ParticleType.Gluon: (0, len(data_file))})
    # saving the file
    np.save("gluon_efps_d5.npy", X)

