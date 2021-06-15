# boid_test.py
# Mathew Titus, 2021
# 
# A test using a simple boid simulation to validate
# the output of the hiton_ezk package.
# 
######################################################


### The Call ###
# from a python environment with working directory at hiton_ezk/src/hiton_ezk/
# enter the following command:
# ```
# exec(open('/Users/mtitus/Documents/GitHub/hiton_ezk/tests/boid_test.py').read())
# ```
import matplotlib.animation as animation

# initialize scripts/functions:
if True:
    # define package elements
    exec(open('hiton_local.py').read())
    # initialize parameter object(s)
    params = define_defaults()
    # get boid tech
    # exec(open('./boid_scripts.py').read())
    exec(open('/Users/mtitus/Documents/GitHub/hiton_ezk/tests/boid_scripts.py').read())

# initialize experiment:
if False:
    # use the build_boid_params method to get a sampler_params object
    spec_vec = np.zeros(10, dtype=int)
    sampler_params = build_boid_params(spec_vec, interaction_radius=10.0)
    # sampler_params['state'] = [[50,50,0],[58,50,0.7],[22,40,1.6]]
    sampler_params['plot'] = False

    # run simulation for NT time steps to get new starting state:
    if True:
        NT = 200
        sampler_params['plot'] = True
        old_duration = sampler_params['duration']
        print("duration was {} -- setting to {}".format(old_duration, NT))
        # sampler_params = simulate(sampler_params | {'duration': NT})
        # sampler_params = sampler_params | {'duration': old_duration}
        sampler_params = simulate(sampler_params.update({'duration': NT}))
        sampler_params = sampler_params.update({'duration': old_duration})
        print("returning duration to {}".format(sampler_params['duration']))
        sampler_params['plot'] = False

    # update the default params object with the values in sampler_param
    # params['sampling_params'] = params['sampling_params'] | sampler_params
    params['sampling_params'] = params['sampling_params'].update(sampler_params)
    state = params['sampling_params']['state']
    specs = params['sampling_params']['species']
    nagents = len(state)

if False:
    params = define_defaults()
    state = params['sampling_params']['state']
    nagents = len(state)

# display supporting parameters on command line:
if False:
    for item in np.setdiff1d(list(params['sampling_params']), ['distributions', 'state']):
        print("{}: {}".format(item, params['sampling_params'][item]))

# calculate ground-truth network:
if False:
    xt = [state[ag][0] for ag in range(nagents)]
    yt = [state[ag][1] for ag in range(nagents)]
    grd_truth = {}
    for ag in range(nagents):
        gt = get_nhbrs(ag, np.array(xt), np.array(yt), params['sampling_params']['int_rad'])
        grd_truth[ag] = np.setdiff1d(gt, ag)

    print("Calculated network: \t\t Ground truth:")
    for ag in range(nagents):
        print("{} \t\t {}".format('???', grd_truth[ag]))

# calculate inferred CPN:
if False:
    # lcn = hiton(0, params['sampling_params']['state'], params)
    nwk = hiton_iterator(params['sampling_params']['state'], params)

    print("Calculated network: \t\t Ground truth:")
    for ag in range(nagents):
        print("{} \t\t {}".format(nwk['tpc'][ag], grd_truth[ag]))

# plot networks w/ quiver:
if False:
    vxt = [np.cos(state[ag][2]) for ag in range(nagents)]
    vyt = [np.sin(state[ag][2]) for ag in range(nagents)]
    # plotting setup
    colors = ['b', 'r']
    species_colors = [colors[sp] for sp in specs]
    plt.ion()
    # plt.clf()
    f, ax = plt.subplots(1,2)
    plt.sca(ax[0])
    ax[0].set_xlim([0, 100.0])
    ax[0].set_ylim([0, 100.0])
    plt.scatter(xt,yt,c=species_colors)
    # plt.quiver(xt,yt,vxt,vyt,width=0.0025)
    for ag in range(nagents):
        qux = []
        quy = []
        quu = []
        quv = []
        for src in grd_truth[ag]: # nwk['tpc'][ag]:
            dx = state[ag][0] - state[src][0]
            dy = state[ag][1] - state[src][1]
            qux.append(state[src][0])
            quy.append(state[src][1])
            quu.append(dx)
            quv.append(dy)
        plt.quiver(qux, quy, quu, quv)#, width=0.25)
    plt.draw()
    plt.pause(0.0001)

# define experiments

try:
    filename = "../../tests/expa.json"
    with open(filename, 'r') as f:
        expa = json.load(f)
except FileNotFoundError:
    expa = {
        'species': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'int_rad': 15.0,
        'int_matrix': [['F']],
        'state': [],
        'name': 'A',
        'frames': []
    }

try:
    filename = "../../tests/expb.json"
    with open(filename, 'r') as f:
        expb = json.load(f)
except FileNotFoundError:
    expb = {
        'species': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'int_rad': 10.0,
        'int_matrix': [['F', 'A'], ['A', 'F']],
        'state': [],
        'name': 'B',
        'frames': []
    }

try:
    filename = "../../tests/expc.json"
    with open(filename, 'r') as f:
        expc = json.load(f)
except FileNotFoundError:
    expc = {
        'species': np.zeros(30, dtype=int).tolist(),
        'int_rad': 8.0,
        'int_matrix': [['F']],
        'state': [],
        'name': 'C',
        'frames': []
    }

try:
    filename = "../../tests/expd.json"
    with open(filename, 'r') as f:
        expd = json.load(f)
except FileNotFoundError:
    expd = {
        'species': np.hstack((np.zeros(20, dtype=int), np.ones(20, dtype=int))).tolist(),
        'int_rad': 8.0,
        'int_matrix': [['F', 'A'], ['A', 'F']],
        'state': [],
        'name': 'D',
        'frames': []
    }

try:
    filename = "../../tests/expe.json"
    with open(filename, 'r') as f:
        expe = json.load(f)
except FileNotFoundError:
    expe = {
        'species': np.hstack((np.zeros(20, dtype=int), np.ones(20, dtype=int))).tolist(),
        'int_rad': 8.0,
        'int_matrix': [['F', 'N'], ['N', 'F']],
        'state': [],
        'name': 'E',
        'frames': []
    }

# Simulate Experiment
trial = 1

def simulate_experiment(exp: dict):
    sampler_params = build_boid_params(
        exp['species'], 
        interaction_matrix=exp['int_matrix'], 
        interaction_radius=exp['int_rad'])
    # generate params dict, needs sampling_params updated
    # sampling_params: state, distributions, simulator, topologies, target_variable...
    params = define_defaults()
    # params['sampling_params'] = params['sampling_params'] | sampler_params
    params['sampling_params'] = params['sampling_params'].update(sampler_params)
    params['sampling_params']['duration'] = 3000
    params['sampling_params']['plot'] = False
    sim_out = simulate(params['sampling_params'], "../../tests/experiment{}_{}.json".format(exp['name'], trial))
    exp['simulation_output'] = sim_out
    params['sampling_params']['duration'] = 1
    exp['params'] = params
    return exp

#### Call to start simulation
# expa = simulate_experiment(expa)
# expb = simulate_experiment(expb)

# User selects frames of interest

def make_video(exp: dict):
    save_path = "../../tests/"
    save_name = save_path + "experiment{}_{}.json".format(exp['name'], trial)
    # load simulation data
    with open(save_name, 'r') as f:
        data = json.load(f)
    # define variables
    x = np.array(data["x"])
    y = np.array(data["y"])
    L = 100.0
    nagents = x.shape[0]
    nt = x.shape[1]
    species = data["species"]
    color_list = np.array(['b','r','g'])
    species_colors = [color_list[species[j]] for j in range(len(species))]
    # create plot object
    f, ax = plt.subplots()
    plt.xlim(0, L)
    plt.ylim(0, L)
    boids = ax.scatter(x[:, 0], y[:, 0], c=species_colors)
    # define animating function
    def animate(t: int):
        xx = x[:, t].flatten()
        yy = y[:, t].flatten()
        plt.title(str(t))
        boids.set_offsets(np.c_[xx, yy])
    # call the animator
    anim = animation.FuncAnimation(f, animate, frames=nt, interval=20)  # init_func=None, blit=True)
    anim_name = save_path + "anim{}_{}.gif".format(exp['name'], trial)
    anim.save(anim_name, writer="imagemagick")

### Call to get video
# make_video(expa)
### Record the good frames
expa['frames'] = [50, 100, 510, 1000]
expb['frames'] = [330, 490, 2990]
expc['frames'] = [48, 170, 278]
expd['frames'] = [49, 125, 195]
expe['frames'] = [195, 350, 1480]
# Calculate CPNs for those frames

def get_exp_cpn(exp: dict):
    save_path = "../../tests/"
    save_name = save_path + "experiment{}_{}.json".format(exp['name'], trial)
    with open(save_name, 'r') as f:
        data = json.load(f)
    for frame in exp['frames']:
        nagents, nt = np.array(data['x']).shape
        state = [[data['x'][ag][frame], data['y'][ag][frame], np.arctan2(data['vy'][ag][frame], data['vy'][ag][frame])] for ag in range(nagents)]
        params = define_defaults()
        sampler_params = build_boid_params(
            exp['species'], 
            interaction_matrix=exp['int_matrix'], 
            interaction_radius=exp['int_rad'])
        new_samp_params = params['sampling_params']
        new_samp_params.update(sampler_params)
        new_samp_params['duration'] = 1
        new_samp_params['plot'] = False
        params['sampling_params'] = new_samp_params
        # print(params)
        # input("experiment's paramaeters, fed into hiton_iterator:\n{}".format(exp['params']))
        cpn = hiton_iterator(state, params)
        with open(save_path + "cpn{}_{}.json".format(exp['name'], trial), 'w+') as f:
            json.dump(cpn, f)

# Calculate CCH for the network

if False:
    nwk = {
        0: [],
        1: [9],
        2: [5],
        3: [5, 7],
        4: [9],
        5: [2, 3],
        6: [],
        7: [3],
        8: [],
        9: [1, 4]
    }
    network = {}
    network['tpc'] = nwk
    make_cch(nwk, plot=False)

# Calculate policies for the clusters (individual level)





# 10 boid trial, at t=200

# Calculated network:          Ground truth:
# [2]           []
# [4, 9]        [9]
# [3, 5]        [5]
# [5, 7]        [5 7]
# [1, 9]        [9]
# [3, 2]        [2 3]
# []            []
# [2, 3]        [3]
# []            []
# [4, 1]        [1 4]



# Clusters:
# 2 - 5 - 3 - 7
# 1 - 9 - 4
# 0
# 6
# 8
