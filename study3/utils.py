RNG = np.random.default_rng(42)

def clip_int(x, lo, hi):
    return int(np.clip(x, lo, hi))