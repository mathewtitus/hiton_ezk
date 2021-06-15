# ezk.py
# Mathew Titus, 2021
# 
# Algorithm for estimating the Wasserstein metric
# 
########################################

import numpy as np
from scipy.optimize import linprog
from hiton_ezk.data_structures import CpnVar, CpnSub


# final argument used to define the distance between points
# for our 1-d cases this is used to calculate the bounds on constraint equations
def estimate_wasserstein(samples, n1, n2, example_var: CpnVar):
    # Define transform matrices
    n: int = n1 + n2
    Ag = np.zeros((n+1,n))
    Bg = np.ones((n,n)) # a "pseudoinverse" of Ag
    # TODO: Calculate Ag & Bg for the R1 topology (formulas below only apply to S1 / T1)
    for i in range(n):
        Ag[i,i] = 1
        Ag[i,np.mod(i+1,n)] = -1
        Ag[n,i] = 1
        for j in range(n):
            Bg[i,np.mod(i+j,n)] = (n-j-1)/n
    # construct objective function
    lf_a = (1.0 / n1) * np.ones((1, n1))
    lf_b = -(1.0 / n2) * np.ones((1, n2))
    linfunct = np.hstack((lf_a, lf_b))[0]
    # reorder sample values and coefficients to ascending order
    ordering = np.argsort(samples)
    samples = samples[ordering]
    f = linfunct[ordering]
    g = np.matmul(Bg.transpose(), f)
    dists = np.zeros(n)
    idists = np.zeros(n)
    # convert to a unit cube with hyperplanar constraint
    if example_var.topology == 'S1':
        for i in range(n):
            j = np.mod(i+1, n)
            d = example_var.dist(samples[i],samples[j])
            dists[i] = d
            idists[i] = 1/d
    elif example_var.topology == 'R1':
        for i in range(n-1):
            j = i+1
            d = example_var.dist(samples[i],samples[j])
            dists[i] = d
            idists[i] = 1/d
    else:
        raise Exception("Error: `estimate_wasserstein` method does not \
            support topology {}!".format(example_var.topology))
    # maximize dot product with this new vector subject to unit cube & constraint dot product
    h = np.matmul(np.diag(dists.flatten()), g)
    # solution alpha must satisfy alpha * dists = 0.
    # find projection of objective vector
    h_tilde = h - (np.dot(h,dists)/np.dot(dists,dists)) * dists
    # arrange system axes so the components reach the boundary in component order
    order_by_mag = np.argsort(-np.abs(h_tilde)) # largest to smallest
    h = h[order_by_mag]
    h_tilde = h_tilde[order_by_mag]
    dists = dists[order_by_mag]
    # sign_vec would be the choice solution within the cube, but violates the orthoganality constraint
    sign_vec = np.sign(h_tilde)
    # working heuristic to avoid saturating too many coordinates leaving no solution set
    cum_alpha_dot_dist = np.abs( np.cumsum(sign_vec * dists) )
    cum_ordered_dists = np.sum(dists) - np.cumsum(dists)
    # find out how many entries we can saturate while retaining a solution set
    # saturated_inds = np.where(cum_ordered_dists < cum_ordered_dists[-1]/2)[0]
    # remaining_inds = np.where(cum_ordered_dists >= cum_ordered_dists[-1]/2)[0] # TODO: replace with np.arange call for speed
    saturated_inds = np.where(cum_alpha_dot_dist < cum_ordered_dists)[0]
    remaining_inds = np.where(cum_alpha_dot_dist >= cum_ordered_dists)[0]
    if len(saturated_inds) == n:
        print("Length autofilled indices: {}\nLength remaining indices: {}\
            \nCalculating final 10\% of entries with linprog".format(len(saturated_inds), len(remaining_inds)))
        saturated_inds = np.array( np.arange(int(np.ceil(0.9 * n))) )
        remaining_inds = np.array( np.setdiff1d(np.arange(n), saturated_inds) )
    # collect saturated coordinate values
    mid_soln = np.zeros(h_tilde.shape)
    mid_soln[saturated_inds] = sign_vec[saturated_inds]
    mid_dp_with_dists = np.dot(mid_soln, dists)
    # use package to fill in remaining values
    h2 = h[remaining_inds]
    d2 = dists[remaining_inds]
    h_t2 = h2.flatten() - (np.dot(h2, d2)/np.dot(d2,d2)) * d2.flatten()
    soln2 = linprog(h_t2, A_eq = d2.reshape((1,-1)), b_eq = mid_dp_with_dists,
        bounds=[(-1,1) for _ in range(len(remaining_inds))])
    # combine
    soln = mid_soln.copy()
    soln[remaining_inds] = -soln2.x
    # get optimal value
    value = np.dot(soln, h)
    # occasionally validate the method
    if np.random.rand() < 1/n:
        val_soln = linprog(h, A_eq = dists.reshape((1,-1)), b_eq = 0,
            bounds=[(-1,1) for _ in range(len(h))])
        print("--- Random Quality Control Call ---\nHeuristic method: \t\t\t{}\nFull linear programming solution:\t{}".format( np.round(value, 2), np.round(-val_soln.fun, 2) ))
    # unwind transformation to get solution vector
    beta_soln = dists * soln
    alpha_soln = np.matmul(Bg, beta_soln.reshape((-1,1)))
    return value, alpha_soln

