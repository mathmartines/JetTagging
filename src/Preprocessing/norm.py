"""File that specifies possible normalizations."""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def norm(n,data):
    # n = 0, 1, 2, 3, 4, 5
    
    if n==0:
        return data
    elif n==1:
        return MinMaxScaler().fit_transform(data)
    elif n==2:
        return StandardScaler().fit_transform(data)
    elif n==3:
        return np.log(data)
    elif n==4:
        return MinMaxScaler().fit_transform(np.log(data))
    else: # n==5
        return StandardScaler().fit_transform(np.log(data))
    
# lembrando que não é possível aplicar MinMax/Standard e depois log, por isso não acrescentamos essas opções.
