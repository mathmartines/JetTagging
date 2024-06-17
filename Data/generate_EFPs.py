"""
Running the EFPs on the full dataset takes a long time to run it all once and store them in a input data, instead
of performing the calculation many times.
"""

from src.Preprocessing import PreprocessingEFPs, create_labels_single_column
from src.Particle import ParticleType
import pandas as pd
import numpy as np


if __name__ == "__main__":
    file_name = "t_jets"

    # Only EFPs with degree <= 5 and fully connected
    efp_processing = PreprocessingEFPs(5, create_labels_single_column, ("p==", 1))

    # Generating the EFT for each dataset
    for dataset in ("Trainning", "Validation", "Test"):
        print(f"Generating EFPs for {dataset}/{file_name}")
        data_file = pd.read_csv(f'{dataset}/{file_name}.csv', header=None)
        # using the Preprocessing to generate the EFPs
        X = efp_processing.transform(X=data_file.to_numpy(), y={ParticleType.Gluon: (0, len(data_file) - 1)})
        # saving the file
        np.save(f"{dataset}/{file_name}_efps_d5_primed.npy", X)

    print("done!")
