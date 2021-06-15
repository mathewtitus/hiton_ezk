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
from typing import List
from hiton_ezk.data_structures import CpnSub, CpnVar
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
    

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


def get_gtn(xt, yt, int_rad: float):
    nagents = len(xt)
    # if isinstance(frame[0], List):
    #     xt = [ag[0] for ag in frame]
    #     yt = [ag[1] for ag in frame]
    # elif isinstance(frame[0], CpnSub):
    #     xt = [ag[0].value for ag in frame]
    #     yt = [ag[1].value for ag in frame]
    gtn = {}
    for ag in range(nagents):
        gt = get_nhbrs(ag, np.array(xt), np.array(yt), int_rad)
        gtn[ag] = np.setdiff1d(gt, ag)
    return gtn


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
    old_axes = plt.gca()
    plt.sca(ax)
    ax.set_xlim([0, 100.0])
    ax.set_ylim([0, 100.0])
    ax.scatter(xt,yt,c=species_colors)
    ax.quiver(xt,yt,vxt,vyt,width=0.0025)
    ax.title.set_text(str(t))
    plt.draw()
    plt.pause(0.0001)
    plt.sca(old_axes)
    # pause, allow for save
    if t%10==0:
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


def simulate(params: dict, save_file: str = ""):
    """
    Simulation Parameters: torus_len, sigma_theta, v_max, int_rad, flock_coupling, repulsion_coupling, style
    """
    verbose = False
    if len(save_file) > 0:
        save_simulation = True
    # unpack parameters
    state = params['state']
    nt = params['duration']
    print("nt = {}".format(nt))
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
            gtn = get_gtn(xmat[:,t], ymat[:,t], int_rad)
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
            params = params | {'state': new_state}
            # plotter(params, t, ax[1])
            # plt.pause(0.01)
            ax[1].cla()
            ax[1].scatter(np.arange(len(state)), [len(gtn[ag]) for ag in gtn])
            plt.draw()
            plt.pause(0.01)
            # plt.close()
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
    if save_simulation:
        sav = {
            'x': xmat.tolist(), 
            'y': ymat.tolist(), 
            'vx': vxmat.tolist(), 
            'vy': vymat.tolist(), 
            'species': smat, 
            'int_matrix': int_matrix, 
            'int_rad': int_rad
        }
        print("sav: {}".format(sav))
        with open(save_file, "w+") as f:
            json.dump(sav, f, indent = 2)
    return params | {'state': new_state}


def build_boid_params(species_vector, interaction_matrix = [['F']], pert_noise = [15, np.pi/6], interaction_radius: float = 5.0, torus_len: float = 100.0):
    number_agents = len(species_vector)
    pert_noise[0] = interaction_radius*2/3
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


def plot_from_save(save_file: str):
    pass


def get_blanket(subset: List, network: dict):
    """
    """
    full_blanket = []
    for ag in subset:
        # get each elements enighbors = MB(ag)
        full_blanket += network[ag]
    blanket_with_reps = np.setdiff1d(full_blanket, subset)
    blanket = np.unique(blanket_with_reps)
    blanket.sort()
    return blanket


def get_cluster(subset: List, network: dict):
    """
    """
    full_blanket = subset.copy()
    if isinstance(full_blanket, np.ndarray):
        print("making full_blanket a list")
        full_blanket = full_blanket.tolist()
    for ag in subset:
        # get each elements enighbors = MB(ag)
        if isinstance(network[ag], np.ndarray):
            print("making network[{}] a list".format(ag))
            network[ag] = network[ag].tolist()
        full_blanket += network[ag]
    blanket = np.unique(full_blanket)
    blanket.sort()
    return blanket.tolist()


def make_cch(network: dict, plot=False):
    """
    """
    cch = {}
    for ag in network:
        hierarchy = [ag]
        last_cluster_size = 1
        cluster = get_cluster([ag], network)
        # hierarchy.append(cluster)
        while len(cluster) != last_cluster_size:
            hierarchy.append(cluster)
            last_cluster_size = len(cluster)
            cluster = get_cluster(cluster, network)
            # hierarchy.append(cluster)
        cch[ag] = hierarchy
        if plot:
            nagents = len(network)
            row = []
            col = []
            for agi in network:
                for agj in network[agi]:
                    row.append(agi)
                    col.append(agj)
            row = np.array(row)
            col = np.array(col)
            data = np.ones(row.shape[0], dtype=int)
            graph = csr_matrix((data, (row, col)), shape=(nagents, nagents))
            # TODO: This dist matrix is dense and includes inf's, needs to change
            dist_matrix = shortest_path(graph, directed=False, unweighted=True)
            Z = linkage(dist_matrix)
            dn = dendrogram(Z)
            plt.show()
    return cch




