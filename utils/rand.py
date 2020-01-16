"""

"utils/rand.py"

Miscellaneous utils.

"""

import numpy as np


# ---------------- CONFIG ----------------
SEED = 10101


# ---------------- RANDOM FUNCTIONS ----------------
def random(a=0, b=1):
    """Random number from range

    :param a: min
    :param b: max
    :returns: random number in [a, b]

    """

    return np.random.rand() * (b - a) + a


def shuffle_with_seed(x, seed=SEED):
    """Randomly shuffles with preset seed and then restores random seed

    :param x: Iterable to be shuffled (not in-place)
    :param seed: seed value (default: random.SEED)
    :returns: shuffled copy of x

    """

    x = np.array(x).copy()

    np.random.seed(seed)
    np.random.shuffle(x)
    np.random.seed(None)

    return x
