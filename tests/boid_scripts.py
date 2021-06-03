# boid_scripts.py
# Mathew Titus, 2020
# 
# 
# 
##################################################################

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def get_nhbrs(focal: int, x_pos: np.ndarray, y_pos: np.ndarray, int_rad: float):
    """
    returns list of influencers of focal boid
    """
    x0 = x_pos[focal]; 
    y0 = y_pos[focal];
    x_dist = x_pos - x0; 
    y_dist = y_pos - y0;
    dists = np.sqrt( x_dist**2 + y_dist**2 )
    nhbrs = np.where( dists < int_rad )[0]
    return nhbrs


def get_interaction(style: str, vars1: dict, vars2: dict):
    """
    function processing boid-boid interaction based on species
    get boid1's change in velocity based on the states of boid1 and boid2
    """
    if style == 'F': # boid aligns
        vx = vars2['vx']
        vy = vars2['vy']
    elif style == 'A': # boid flees
        dx = vars1['x'] - vars2['x']
        dy = vars1['y'] - vars2['y']
        offset = np.sqrt(dx**2 + dy**2)
        vx = dx / offset
        vy = dy / offset
    elif style == 'N': # boid don't notice
        vx = 0
        vy = 0
    else:
        raise Exception("In get_interaction: {} is not a valid flocking style.".format(style))
    return vx, vy


def plotter(params, t, ax):
    # collect variables
    state = params['state']
    specs = params['species']
    nagents = len(state)
    xt = [state[ag][0] for ag in range(nagents)]
    yt = [state[ag][1] for ag in range(nagents)]
    vxt = [np.cos(state[ag][2]) for ag in range(nagents)]
    vyt = [np.sin(state[ag][2]) for ag in range(nagents)]
    # plotting setup
    colors = ['b', 'r']
    species_colors = [colors[sp] for sp in specs]
    # plt.clf()
    # plt.sca(ax[0])
    # ax[0].set_xlim([0, 100.0])
    # ax[0].set_ylim([0, 100.0])
    # plt.scatter(xt,yt,c=species_colors)
    # plt.quiver(xt,yt,vxt,vyt,width=0.0025)
    # plt.title(str(t))
    # plt.draw()
    # plt.pause(0.0001)
    plt.sca(ax)
    ax.set_xlim([0, 100.0])
    ax.set_ylim([0, 100.0])
    ax.scatter(xt,yt,c=species_colors)
    ax.quiver(xt,yt,vxt,vyt,width=0.0025)
    ax.title.set_text(str(t))
    plt.draw()
    plt.pause(0.0001)
    if t%10==9:
        filename = input("Press enter to continue.\nEnter a filename to save the simulation state.")
        if filename != "":
            save_dict = {
                'state': params['state'],
                'species': params['species'],
                'topologies': params['topologies'],
                'int_matrix': params['int_matrix'],
                'int_rad': params['int_rad'],
                'sigma_theta': params['sigma_theta'],
                'v_max': params['v_max'],
                'torus_len': params['torus_len'],
                'target_variable': params['target_variable'],
                'duration': params['duration'],
                'plot': True,
            }
            with open("./"+filename+".json", 'w+') as f:
                json.dump(save_dict, f, indent=2)
    # plt.close()


def simulate(params: dict):
    """
    Simulation Parameters: torus_len, sigma_theta, v_max, int_rad, flock_coupling, repulsion_coupling, style
    """
    verbose = False
    # unpack parameters
    state = params['state']
    nt = params['duration']
    int_matrix = params['int_matrix']
    int_rad = params['int_rad']
    v_max = params['v_max']
    sigma_theta = params['sigma_theta']
    torus_len = params['torus_len']
    # initialize species matrix
    smat = params['species']
    nagents: int = len(state)
    # locations of each agent
    xmat = np.zeros((nagents, nt+1))
    ymat = np.zeros((nagents, nt+1))
    # velocities of each agent
    vxmat = np.zeros((nagents, nt+1))
    vymat = np.zeros((nagents, nt+1))
    # set up initial conditions: random directions and position
    xmat[:, 0] = [k[0] for k in state] #x0
    ymat[:, 0] = [k[1] for k in state] #y0
    vxmat[:, 0] = [np.cos(k[2]) for k in state] #vx0
    vymat[:, 0] = [np.sin(k[2]) for k in state] #vy0
    # begin the loop
    # if params['plot']: f, ax = plt.subplots(1,2)
    if verbose: print("simulator input state:\n{}".format(state))
    for t in range(1, nt+1):
        for k in range(nagents):
            # update agent locations
            xmat[k, t] = np.mod(xmat[k, t-1] + vxmat[k, t-1], torus_len)
            ymat[k, t] = np.mod(ymat[k, t-1] + vymat[k, t-1], torus_len)
            # update the velocities of the agents
            nhbrs = get_nhbrs(k, xmat[:, t-1], ymat[:, t-1], int_rad)
            vxmat[k,t] = vxmat[k,t-1] + np.random.randn() / np.sqrt(2.0) * sigma_theta
            vymat[k,t] = vymat[k,t-1] + np.random.randn() / np.sqrt(2.0) * sigma_theta
            for nhbr in np.setdiff1d(nhbrs, k):
                kay_state = {
                    'x': xmat[k, t-1],
                    'y': ymat[k, t-1],
                    'vx': vxmat[k, t-1],
                    'vy': vymat[k, t-1]
                }
                nhbr_state = {
                    'x': xmat[nhbr, t-1],
                    'y': ymat[nhbr, t-1],
                    'vx': vxmat[nhbr, t-1],
                    'vy': vymat[nhbr, t-1]
                }
                dvx, dvy = get_interaction(int_matrix[smat[k]][smat[nhbr]], kay_state, nhbr_state)
                vxmat[k,t] += dvx
                vymat[k,t] += dvy
            v_abs = np.sqrt(vxmat[k,t]**2 + vymat[k,t]**2)
            v_target = v_max
            vxmat[k,t] = vxmat[k,t] / v_abs * v_target
            vymat[k,t] = vymat[k,t] / v_abs * v_target
        if (params['plot']):
            plt.ion()
            if 'f' in dir():
                f.clf()
                ax = f.subplots(1,2)
            else:
                f, ax = plt.subplots(1,2)
            plotter(params, t-1, ax[0])
            new_state = [
                [
                    xmat[k,t], 
                    ymat[k,t], 
                    np.arctan2(vymat[k,t], vxmat[k,t])
                ] 
                for k in range(nagents)
            ]
            plotter(params | {'state': new_state}, t, ax[1])
            plt.close()
    # gather final state of system
    new_state = [
        [
            xmat[k,-1], 
            ymat[k,-1], 
            np.arctan2(vymat[k,-1], vxmat[k,-1])
        ] 
        for k in range(nagents)
    ]
    if verbose: print("simulator output state:\n{}".format(new_state))
    return params | {'state': new_state}


def build_boid_params(species_vector, interaction_matrix = [['F']], pert_noise = [15, 3], interaction_radius: float = 5.0, torus_len: float = 100.0):
    number_agents = len(species_vector)
    default_state = []
    default_dists = []
    default_topos = []
    default_specs = []
    for ag in range(number_agents):
        x0 = np.random.uniform.__call__(**{'low': 0.0, 'high': torus_len})
        y0 = np.random.uniform.__call__(**{'low': 0.0, 'high': torus_len})
        theta0 = np.random.uniform.__call__(**{'low': -np.pi, 'high': np.pi})
        default_state.append([x0, y0, theta0])
        # default_dists.append([
        #     {'variable': np.random.normal, 'params': {'low': 0.0, 'high': torus_len}, 'epsilon': pert_noise[0] },
        #     {'variable': np.random.normal, 'params': {'low': 0.0, 'high': torus_len}, 'epsilon': pert_noise[0] },
        #     {'variable': np.random.normal, 'params': {'low': -np.pi, 'high': np.pi}, 'epsilon': pert_noise[1] }
        # ])
        default_dists.append([
            {'variable': np.random.uniform, 'params': {'low': 0.0, 'high': torus_len}, 'epsilon': pert_noise[0] },
            {'variable': np.random.uniform, 'params': {'low': 0.0, 'high': torus_len}, 'epsilon': pert_noise[0] },
            {'variable': np.random.uniform, 'params': {'low': -np.pi, 'high': np.pi}, 'epsilon': pert_noise[1] }
        ])
        default_topos.append([
            {'T1': torus_len},
            {'T1': torus_len},
            {'S1': np.nan}
        ])

    param_dict = {
        'state': default_state,
        'simulator': simulate,
        'distributions': default_dists,
        'topologies': default_topos,
        'perturbation': False,
        'target_variable': [0,2],
        'duration': 1,
        # boid specific parameters
        'plot': True,
        'int_rad': interaction_radius,
        'int_matrix': interaction_matrix,
        'v_max': 0.50,
        'sigma_theta': 0.05,
        'torus_len': torus_len,
        'species': species_vector
    }
    return param_dict


def load_params(filename: str):
    if filename.index('.json') < 0: filename += ".json"
    assert(os.exists(filename)), "File {} does not exist.".format(filename)
    params = {}
    with open(filename, 'r') as f:
        params = json.load(f)
    params['simulator'] = simulate
    params['perturbation'] = False
    nagents = len(params['state'])
    default_dists = []
    for ag in range(number_agents):
        default_dists.append([
            {'variable': np.random.normal, 'params': {'low': 0.0, 'high': torus_len}, 'epsilon': pert_noise[0] },
            {'variable': np.random.normal, 'params': {'low': 0.0, 'high': torus_len}, 'epsilon': pert_noise[0] },
            {'variable': np.random.normal, 'params': {'low': -np.pi, 'high': np.pi}, 'epsilon': pert_noise[1] }
        ])
    params['distributions'] = default_dists
    return params



