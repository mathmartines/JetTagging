"""Quick script to find the corresponding graphs for the EFPs"""

from src.Preprocessing.JetPreprocessing import PreprocessingEFPs

if __name__ == "__main__":
    # Only EFPs with degree <= 5 and fully connected
    efp_processing = PreprocessingEFPs(5,  ("p==", 1))
    efp_set = efp_processing.efps_set

    # information about each polynomial
    index = 2     # column index of the EFP dataset
    graph = efp_set.graphs(index)

    n, _, d, v, _, c, p, _ = efp_set.specs[index]

    print("Graph:", graph)
    print("Number of vertices, n:", n)
    print("Number of edges,    d:", d)
    print("Maximum valency,    v:", v)
    print("VE complexity,      c:", c)
    print("Number of primes,   p:", p)
