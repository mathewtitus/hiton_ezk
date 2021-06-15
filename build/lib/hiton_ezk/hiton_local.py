# hiton.py
# Mathew Titus, 2021
# 
# 
# 
################################################################

import sys
import numpy as np
from typing import List
from ezk import estimate_wasserstein
from helper import powerset, c_subselect, sigmoid
from data_structures import CpnVar, CpnSub

choleskySetting = True # default: True
symPosSetting = False # default: True


def sample_function(params: dict):
    """
    An example of a simulation function format that is
    compatible with the sample generation needed for EZK
    dependence measure estimation.

    The params dictionary holds all the information needed
    to initialize and simulate the system. A perturbing random
    variable is needed for each variable in `state` to
    perform sampling; this is stored in `distributions`.

    It is recommended that the duration of simulation be one 
    of the parameters for tuning the time between variable 
    perturbation and focal variable response.
    """
    # print("Calling sampler with state\n{}".format(params['state']))
    # if params['perturbation']:
    #     print("Calling sample_function with perturbation; variables \
    #         are sampled by random variables of the form: {}".format([a.__name__ for a in params['distributions']]))
    # if 'duration' in params:
    #     print("Executing simulation for {} time steps.".format(params['duration']))
    # # presumably state should change; sample_function just passes state back into params unchanged
    # return params | {'state': params['state']}
    # return random state
    new_state = []
    for ag in params['state']:
        # new_state.append(np.random.randn(len(ag)).tolist())
        new_state.append(ag)
    # print("{}\n".format(new_state))
    return params | {'state': new_state}


def rlt_dist(v1: CpnVar, v2: CpnVar):
    if v1.topology == "R1":
        return v1.dist(v1.perturbation, v2.perturbation)
    elif v1.topology == "S1":
        return v1.dist(v1.perturbation, v2.perturbation) / (2 * np.pi)
    elif v2.topology == "T1":
        return v1.dist(v1.perturbation, v2.perturbation) / v1.length
    else:
        raise Exception("Not an ok topology ({}). Tossing an error.".format(v1.topology))


def l2_agent_distance(agent1: CpnSub, agent2: CpnSub):
    sq_dist = 0.0
    # sum squared differences
    for v in range(len(agent1)):
        sq_dist += rlt_dist(agent1[v], agent2[v])**2 
    # return the L2 norm of the two vectors w.r.t. the different components metrics
    return np.sqrt(sq_dist)

# NB / TODO:
# In the CpnSub definitions, the entries are treated as
# random variables which have a perturb method that adds
# another, differently distributed, random variable to the
# CpnVar value. In the below we really only need the 
# perturbation distribution, and this is what should be 
# included in the `sampling_params` dictionary.
def define_defaults():
    default_sampling_params = {
        'state': [
            [0, 0], 
            [0, 1]#, 
#            [1, 0], 
#            [1, 1]
        ],
        'simulator': sample_function,
        'distributions': [
            [
                {
                    'variable': np.random.normal,
                    'params': {'loc': 0, 'scale': 0.5},
                    'epsilon': 0.5
                },
                {
                    'variable': np.random.normal,
                    'params': {'loc': 0, 'scale': 0.5},
                    'epsilon': 0.5
                }
            ],
            [
                {
                    'variable': np.random.normal,
                    'params': {'loc': 0, 'scale': 0.5},
                    'epsilon': 0.5
                },
                {
                    'variable': np.random.normal,
                    'params': {'loc': 0, 'scale': 0.5},
                    'epsilon': 0.5
                }
            ]#,
            # [
            #     {
            #         'variable': np.random.normal,
            #         'params': {'loc': 0, 'scale': 0.5},
            #         'epsilon': 0.5
            #     },
            #     {
            #         'variable': np.random.normal,
            #         'params': {'loc': 0, 'scale': 0.5},
            #         'epsilon': 0.5
            #     }
            # ],
            # [
            #     {
            #         'variable': np.random.normal,
            #         'params': {'loc': 0, 'scale': 0.5},
            #         'epsilon': 0.5
            #     },
            #     {
            #         'variable': np.random.normal,
            #         'params': {'loc': 0, 'scale': 0.5},
            #         'epsilon': 0.5
            #     }
            # ]
        ],
        'topologies': [
            [
                {'R1': np.nan},
                {'T1': 1}
            ],
            [
                {'R1': np.nan},
                {'T1': 1}
            ]#,
            # [
            #     {'R1': np.nan},
            #     {'T1': 1}
            # ],
            # [
            #     {'R1': np.nan},
            #     {'T1': 1}
            # ]
        ],
        'perturbation': False,
        'target_variable': [0,0], # [agent index, variable index]
        'duration': 1
    }
    # create default values to be overwritten with user inputs
    N1 = 200
    N2 = 200
    default_params = {
        'number_per_inclusion': 1,
        'N': 30,
        'N1': N1,
        'N2': N2,
        'minimum_source_agent_distance': 0.2, # 8/np.sqrt(N1 + N2),
        'causal_threshold': 1.0,
        'c_transform': sigmoid,
        'agent_distance': l2_agent_distance,
        'sampling_params': default_sampling_params
    }
    return default_params


def eliminate(x, y, TPC, system, params):
    """
    Use the EZK dependence measure to determine whether
    variable x is independent of y for some subset of TPC.
    This is the elimination step of the HITON interleaving
    algorithm.

    `x` is the single variable that we are interested in learning the causal influences of. 
    It is passed in as a list [x0, x1] with `x0` the member index and `x1` the variable index.

    `y` is an integer, indexing the source member potentially affecting variable `x`.

    `system` comes in as a list of agent states, where the 
    agent state is described by an ordered list of values.
    `cpnsub_system` is a list of CpnSub objects with values
    copied in from `system`, allowing us to make perturb and 
    dist calls.
    """
    verbose = False
    visual = False
    # assert((x not in TPC) & (y not in TPC)), "Error: source or target agents (x_i = {}, x_j = {}) found in TPC = {}.".format(x,y,TPC)
    assert(y not in TPC), "Error: source agent (x_j = {}) found in TPC = {}.".format(y,TPC)
    simulator = params['sampling_params']['simulator']
    # TODO: here we set the target variable based on input parameters... this should either not change or not be part of the params dictionary.
    params['sampling_params']['target_variable'][0] = x
    if verbose: print("calling eliminate to check I({}, {} | {}).".format(x,y,TPC)) # ({},{})\nTPC: {}".format(x,y,TPC))#,params['sampling_params']['target_variable']))
    # convert the lists of values in `system` to a CpnSub object
    cpnsub_system = []
    for agent_ind in range(len(system)):
        cpn_sub = CpnSub([])
        for var_ind in range(len(system[agent_ind])):
            cpn_var = CpnVar(np.random.uniform, {'low': 0.0, 'high': 1.0}, params['sampling_params']['topologies'][agent_ind][var_ind])
            cpn_var.set_val(system[agent_ind][var_ind])
            cpn_sub.append(cpn_var)
        cpnsub_system.append(cpn_sub)
    # generate samples of entire unconditioned system
    samples = []
    sample_pairs = []
    resamples_required: int = 0
    while len(sample_pairs) < params['N']:
        # copy the system vars
        new_sam = [cpn_sub.copy() for cpn_sub in cpnsub_system]
        if verbose & (len(sample_pairs) < 2):
            print("Checking new_sam is a good copy of cpnsub_system:")
            print("CSS vals: {}\nNS vals:  {}\n".format([css.values_to_list() for css in cpnsub_system], [ns.values_to_list() for ns in new_sam]))
            print("CSS perts: {}\nNS perts:  {}\n".format([css.perts_to_list() for css in cpnsub_system], [ns.perts_to_list() for ns in new_sam]))
        # perturb each system var
        for sub_ind in range(len(new_sam)):
            cpn_sub = new_sam[sub_ind]
            for mem_ind in range(len(cpn_sub)):
                member = cpn_sub[mem_ind]
                member.perturb(
                    params['sampling_params']['distributions'][sub_ind][mem_ind]['variable'],
                    params['sampling_params']['distributions'][sub_ind][mem_ind]['epsilon'] 
                )
        if verbose & (len(sample_pairs) < 2):
            print("CSS vals: {}\nNS vals:  {}\n".format([css.values_to_list() for css in cpnsub_system], [ns.values_to_list() for ns in new_sam]))
            print("CSS perts: {}\nNS perts:  {}\n".format([css.perts_to_list() for css in cpnsub_system], [ns.perts_to_list() for ns in new_sam]))
        samples.append(new_sam)
        # record which older samples new_sam can be paired with
        if len(samples) == 1:
            continue
        else:
            for old_sam_ind in range(len(samples)-1):
                old_sam_y = samples[old_sam_ind][y]
                new_sam_y = new_sam[y]
                d_y1_y2 = params['agent_distance'](new_sam_y, old_sam_y)
                if d_y1_y2 >= params['minimum_source_agent_distance']:
                    sample_pairs.append((old_sam_ind, len(samples)-1))
                else:
                    resamples_required += 1
                    if resamples_required % 1000 == 0: 
                        print("{} resamples performed so far for x={}, y={}. Consider raising N1 and N2 parameters.".format(resamples_required,x,y))
    # list all mediating subsets of TPC
    tpc_subsets = powerset(TPC)
    # test I(x, y | Z) for each subset Z
    for sub in tpc_subsets:
        # for each sample pair, generate conditions K, sample from mu_i distributions, and calculate c_xy^K
        c_xyK = []
        for pair in sample_pairs:
            if verbose: print("using sample pair ({}, {})".format(pair[0], pair[1]))
            # samples to calculate W_d^{(k,l)} with
            sample_kay = samples[pair[0]]
            sample_ell = samples[pair[1]]
            # collect variable conditions
            sub_conds_kay = []
            sub_conds_ell = []
            # sample_x is a list of CpnSub's so each needs to be conditioned in seq
            conditioned_cpnsub_kay = []
            conditioned_cpnsub_ell = []
            # calculate distance between j^th agents for later
            d_y1_y2 = params['agent_distance'](sample_kay[y], sample_ell[y])
            if verbose:
                print("sample_kay__________\nvals: {}\nperts: {}".format([sk.values_to_list() for sk in sample_kay], [sk.perts_to_list() for sk in sample_kay]))
                print("sample_ell__________\nvals: {}\nperts: {}".format([sl.values_to_list() for sl in sample_ell], [sl.perts_to_list() for sl in sample_ell]))
                print("dist: {}\n".format(d_y1_y2))
            for agent_ind in range(len(system)):
                if agent_ind in sub:
                    sub_conds_kay.append(sample_kay[agent_ind].perts_to_list())
                    sub_conds_ell.append(sample_kay[agent_ind].perts_to_list())
                elif agent_ind == y:
                    sub_conds_kay.append(sample_kay[agent_ind].perts_to_list())
                    sub_conds_ell.append(sample_ell[agent_ind].perts_to_list())
                else:
                    sub_conds_kay.append([np.nan for _ in range(len(sample_kay[agent_ind]))])
                    sub_conds_ell.append([np.nan for _ in range(len(sample_kay[agent_ind]))])
                # add a copy of agent to conditioned samples
                conditioned_cpnsub_kay.append(sample_kay[agent_ind].copy())
                conditioned_cpnsub_ell.append(sample_ell[agent_ind].copy())
                # perform conditioning
                conditioned_cpnsub_kay[agent_ind] = conditioned_cpnsub_kay[agent_ind].condition(sub_conds_kay[agent_ind], params['sampling_params']['distributions'][agent_ind])
                conditioned_cpnsub_ell[agent_ind] = conditioned_cpnsub_ell[agent_ind].condition(sub_conds_ell[agent_ind], params['sampling_params']['distributions'][agent_ind])
            if verbose:
                print("conditions:\nk: {}\nl: {}\n".format(sub_conds_kay, sub_conds_ell))
                print("conditioned_cpnsub_kay__________\nvals: {}\nperts: {}\n".format([sk.values_to_list() for sk in conditioned_cpnsub_kay], [sk.perts_to_list() for sk in conditioned_cpnsub_kay]))
                print("conditioned_cpnsub_ell__________\nvals: {}\nperts: {}\n".format([sl.values_to_list() for sl in conditioned_cpnsub_ell], [sl.perts_to_list() for sl in conditioned_cpnsub_ell]))
            # initialize samples of variable x
            x_draws = np.zeros(params['N1'] + params['N2'])
            # 
            focal_agent = params['sampling_params']['target_variable'][0]
            focal_var = params['sampling_params']['target_variable'][1]
            # draw from \mu_{x}(z^k_{K u y)
            if visual:
                spc_changes = []
                ang_changes = []
            kay_perts = []
            for k1 in range(params['N1']):
                for mem_ind in range(len(conditioned_cpnsub_kay)):
                    for var_ind in range(len(conditioned_cpnsub_kay[mem_ind])):
                        # perform perturbation of conditioned system
                        if visual: a0 = conditioned_cpnsub_kay[mem_ind][var_ind].value
                        conditioned_cpnsub_kay[mem_ind][var_ind].perturb( 
                            params['sampling_params']['distributions'][mem_ind][var_ind]['variable'], 
                            params['sampling_params']['distributions'][mem_ind][var_ind]['epsilon'] 
                        )
                        if (mem_ind == focal_agent)&(var_ind == focal_var):
                            kay_perts.append(conditioned_cpnsub_kay[mem_ind][var_ind].perturbation)
                        if visual: 
                            if var_ind == 2:
                                ang_changes.append(conditioned_cpnsub_kay[mem_ind][var_ind].perturbation - a0)
                            else:
                                spc_changes.append(conditioned_cpnsub_kay[mem_ind][var_ind].perturbation - a0)
                # generate new simulation results from the new sample 
                sys_kay = [cpn_j.perts_to_list() for cpn_j in conditioned_cpnsub_kay]
                if verbose & (k1 == 5): print("going into simulator: {}".format(sys_kay))
                new_mu = simulator(params['sampling_params'] | {
                    'state': [cpn_sub.perts_to_list() for cpn_sub in conditioned_cpnsub_kay], 
                    'perturbation': True
                })
                x_draws[k1] = new_mu['state'][focal_agent][focal_var]
                if verbose & (k1 == 5): print("coming out of simulator: {}\nx_draw: {}".format(new_mu['state'], x_draws[k1]))
            if visual:
                g, bx = plt.subplots(1,2)
                plt.sca(bx[0])
                plt.hist(spc_changes, bins=15)
                plt.sca(bx[1])
                plt.hist(ang_changes, bins=15)
                plt.draw()
                plt.show()
                input("")
                plt.close()
            # draw from \mu_{x}(z^ell_{K u y)
            ell_perts = []
            for k2 in range(params['N2']):
                for mem_ind in range(len(conditioned_cpnsub_ell)):
                    for var_ind in range(len(conditioned_cpnsub_ell[mem_ind])):
                        # perform perturbation of conditioned system
                        conditioned_cpnsub_ell[mem_ind][var_ind].perturb( 
                            params['sampling_params']['distributions'][mem_ind][var_ind]['variable'], 
                            params['sampling_params']['distributions'][mem_ind][var_ind]['epsilon'] 
                        )
                        if (mem_ind == focal_agent)&(var_ind == focal_var):
                            ell_perts.append(conditioned_cpnsub_ell[mem_ind][var_ind].perturbation)
                # generate new simulation results from the new sample 
                new_mu = simulator(params['sampling_params'] | {
                    'state': [cpn_sub.perts_to_list() for cpn_sub in conditioned_cpnsub_ell], 
                    'perturbation': True
                })
                x_draws[params['N1'] + k2] = new_mu['state'][focal_agent][focal_var]
            # calculate dependence measure
            W_kl, _ = estimate_wasserstein(x_draws, params['N1'], params['N2'], cpnsub_system[focal_agent][focal_var])
            c_xyK.append(W_kl / d_y1_y2)
            if visual:
                plt.ion()
                _, ax = plt.subplots(1,2)
                # ax[0].hist(c_xyK)
                ax[0].hist([kay_perts, ell_perts])
                n = ax[1].hist([x_draws[:params['N1']], x_draws[params['N1']:]], bins=np.linspace(-np.pi,np.pi - np.pi/6,11))
                ax[1].title.set_text("c: {}\ndist: {}\nDiff: {}".format(
                    np.round(c_xyK[-1], 4),
                    np.round(d_y1_y2, 4),
                    np.sum(np.abs(n[0][0] - n[0][1]))/np.sum(n[0][0])
                    )
                )
                plt.draw()
                input("Hit enter for next histogram")
                plt.close()
        # # adjust, removing outliers [OPTIONAL]
        # c_xyK = c_subselect(c_xyK)
        if sub == []: c = c_xyK
        # determine whether I(x, y | Z)
        print("Finished testing I({}, {} | {})\nc_ij^K: {}\ncausal threshold: {}\n".format(x,y,sub,np.max(c_xyK),params['causal_threshold']))
        if np.max(c_xyK) < params['causal_threshold']:
            return True, sub, c_xyK
        else:
            continue
    # if I(X,Y|Z) for no subset Z, return False (do not eliminate from TPC)
    return False, [], c


def hiton(focal_index: int, system: List[List], params: dict):
    """
    Apply the HITON interleaving algorithm of Aliferis, Statnikov, et al. (2010)
    to discover the local causal neighborhood of the `focal_index`th agent.
    The function also returns the results of each I(X, Y | Z) calculation;
    blocking_set[y] is a dictionary of { 'Z': c^Z_{x,y} } pairs.
    """
    assert (focal_index < len(system)), "Focal index ({}) must be less than system size ({}).".format(focal, len(system))
    print("Building LCN for agent {}.".format(focal_index))
    # prepare parameter data
    params = define_defaults() | params
    # initializing...
    nagents = len(system)
    blocking_set = {}
    for k in range(nagents): blocking_set[k] = {}
    ## initialize TPC and OPEN, the list of candidate variables
    TPC = []
    # self at time t-1 is a large influence, need to condition on it to get 
    # a reasonable measurement of the distribution at time t
    OPEN = np.setdiff1d(np.arange(nagents), [focal_index])
    # OPEN = np.arange(nagents)
    # calculated dependence of focal agent on each other agent with Z = []
    unconditioned_dependence = np.zeros(len(OPEN))
    keep_agent = np.zeros(len(OPEN))
    for j in range(len(OPEN)):
        jay = OPEN[j]
        remove_agent, _, c_ijK = eliminate(focal_index, jay, [], system, params)
        unconditioned_dependence[j] = np.max(c_ijK)
        blocking_set[jay]['null'] = c_ijK
        keep_agent[j] = not remove_agent
    # order OPEN by influence (descending)
    open_order = np.flip(np.argsort(unconditioned_dependence))
    OPEN = OPEN[open_order]
    keep_agent = np.bool_(keep_agent[open_order])
    # discard agents which are not causes of the focal agent
    OPEN = list(OPEN[keep_agent])
    # perform interleaving strategy
    # TODO: record this while loop activity in a report that can be printed at the end of execution
    while len(OPEN) > 0:
        print("OPEN: {}\nTPC: {}".format(OPEN, TPC))
        # INCLUSION STEP: transfer most likely agent(s) from OPEN to TPC
        for ns in range(params['number_per_inclusion']): # (eliminate only every number_per_inclusion additions)
            if len(OPEN) > 0:
                # add vertices to the front so new guys get tested for causal dependence first in elimination step
                TPC.reverse()
                TPC.append(int(OPEN[0]))
                TPC.reverse()
                agent_to_tpc = OPEN.pop(0)
        # ELIMINATION STEP: test for independence 
        for agent in TPC:
            remove_agent, Z, c_ijK = eliminate(focal_index, agent, np.setdiff1d(TPC, agent), system, params)
            # record
            Z.sort()
            blocking_set[agent][str(Z)] = c_ijK
            # eliminate
            if remove_agent:
                # TODO: Keep removed agents in separate dictionary, recording their c_ijK values
                TPC.remove(agent)
    print("The final set TPC for agent {} is\n{}".format(focal_index, TPC))
    return TPC, blocking_set


def hiton_iterator(system: List[List], params: dict):
    """
    Call HITON-EZK for each member of the system, results are returned 
    in a dictionary with the parameters used in simulation and the 
    associated blocking variable set (subset K of TPC minimizing c_{i,j}^K)

    `params` holds the parameters for executing the EZK measure approximation.
    `sampling_params` holds the information needed for sampling function to execute;
    in the calls to `eliminate` we will pass samples generated by calling 
    sampling_function(sampling_parameters)

    'sampling_params' has X special key-value pairs that will be defined later including 
    'conditions' which contains a list of lists in the same format as `system`
    """
    assert ((sys.version_info.major == 3) & (sys.version_info.minor >= 9)), "System error: Update Python to a version >= 3.9.1"
    params = define_defaults() | params
    network = {'params': params, 'tpc': {}, 'blocking_set': {}}
    nagents = len(system)
    for focus in np.arange(nagents, dtype=int):
        tpc, Z = hiton(focus, system, params)
        network['tpc'][focus] = tpc
        network['blocking_set'][focus] = Z
    return network


def symmetrize(network):
    """
    Symmetrize the network by ensuring X_i in TPC_j iff X_j in TPC_i.
    This is not used for dynamical simulation where the focal variable 
    X_i cannot influence X_j (as it occurs afterward).
    """
    nagents = len(network['tpc'])
    network['pc'] = {}
    # For each agent
    for aye in range(nagents):
        aye_pc = network['tpc'][aye].copy()
        # Check each potential parent/child
        for jay in network['tpc'][aye]:
            if aye == jay: continue
            # To see if the focal agent is also their parent/child
            if aye not in network['tpc'][jay]:
                aye_pc.remove(jay)
        network['pc'][aye] = aye_pc
    return network







