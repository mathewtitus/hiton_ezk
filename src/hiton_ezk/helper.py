# helper.py
# Mathew Titus, 2021
# 
# 
# 
#################################

from numpy import min, exp, array, zeros, round, mean
from itertools import combinations


def powerset(original_set, subset_size_limit: int = 5):
    """
    Return the set of all subsets of original_set.
    We omit subsets with size greater than subset_size_limit
    to keep the computational complexity fixed, though users
    may want to adjust this depending on the simulated system
    and computing resources.
    """
    s = list(original_set)
    lim = min((subset_size_limit, len(s))) + 1 
    ps = []
    for r in range(lim): ps.extend([list(d) for d in combinations(s,r)])
    ps.reverse()
    return ps


def sigmoid(value):
    """
    """
    value = array(value)
    s = zeros(value.shape)
    # general form of the sigmoidal function
    sigm = 1*exp(4 * (value - 1/2)) / (1 + exp(4 * (value - 1/2)))
    # the y-value of the corrective line at x=0
    y0 = -1*exp(-2) / (1 + exp(-2))
    # the y-value of the corrective line at x=0
    y1 = 1 - exp(2)/(1 + exp(2))
    interval0 = np.where(value < 0.0)
    interval1 = np.where((value >= 0.0) & (value <= 1.0))
    interval2 = np.where(value > 1.0)
    s[interval0] = 0.0
    s[interval1] = sigm[interval1] + (y1-y0)*value[interval1] + y0
    s[interval2] = sigm[interval2] + y1
    return s


def c_subselect(c_K):
    """
    """
    c_K = np.array(list(c_K.values()))
    msrs = c_K.flatten()[c_K.flatten() > 0]
    msrs.sort()
    return mean(msrs[int(round(0.9*len(msrs))):])


