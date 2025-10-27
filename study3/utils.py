from typing import Any

import numpy as np
from numpy import ndarray, dtype, float64

RNG = np.random.default_rng(42)

def clip_int(x, lo, hi):
    return int(np.clip(x, lo, hi))

def jas(pre: float,
        post: float,
        rating: float) -> float:

    """
    Judge Advisors Systems rating

    :param pre: Score before intervention
    :param post: Score after intervention
    :param rating: intervention rating
    :return: JAS rating
    """

    jas = (post - pre) / (rating - pre)

    return jas

def change_distance(pre: float,
                    post: float,
                    rating: float) -> float:

    """
    L1 distance of change

    :param pre: Score before intervention
    :param post: Score after intervention
    :param rating: intervention rating
    :return: JAS rating
    """

    delta = np.abs(rating - pre) / np.abs(rating - post)

    return delta

def sample_beta(a: float,
                b: float,
                loc: float,
                scale: float,
                size: int = 1) -> ndarray[tuple[Any, ...], dtype[float64]]:

    rng = np.random.default_rng(16)
    sample = loc + scale * rng.beta(a, b, size=size)

    return sample

def fit_betabinom(n, mean, var, eps=1e-12):

    p = mean / n
    if not (0 < p < 1):
        raise ValueError("Mean must be strictly between 0 and n.")

    binom_var = n * p * (1 - p)
    t = var / (binom_var + eps)

    if t < 1 - 1e-9:
        raise ValueError("Variance too small for Beta–Binomial (t < 1).")
    if t <= 1 + 1e-9:
        # Binomial limit: kappa -> infinity; approximate with a large kappa
        kappa = 1e12
    else:
        kappa = (n - t) / (t - 1)
        if kappa <= 0:
            # numerically, very large overdispersion (kappa ~ 0). Clip at small positive.
            kappa = 1e-9

    alpha = p * kappa
    beta  = (1 - p) * kappa
    return alpha, beta




