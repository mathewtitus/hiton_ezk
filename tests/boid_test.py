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

# define package elements
exec(open('hiton_local.py').read())

# initialize parameter object(s)
params = define_defaults()

# get boid tech
# exec(open('./boid_scripts.py').read())
exec(open('/Users/mtitus/Documents/GitHub/hiton_ezk/tests/boid_scripts.py').read())

# use the build_boid_params method to get a sampler_params object
spec_vec = np.zeros(3, dtype=int)
sampler_params = build_boid_params(spec_vec, interaction_radius=10.0)
sampler_params['state'] = [[50,50,0],[58,50,0.7],[42,40,1.6]]

# run simulation for NT time steps to get new starting state
NT = 10
if False:
    old_duration = sampler_params['duration']
    print("duration was {} -- setting to {}".format(old_duration, NT))
    sampler_params = simulate(sampler_params | {'duration': NT})
    sampler_params = sampler_params | {'duration': old_duration}
    print("returning duration to {}".format(sampler_params['duration']))

# turn off plotting
sampler_params['plot'] = False
print("turning plotting to {}".format(sampler_params['plot']))

# update the default params object with the values in sampler_param
params['sampling_params'] = params['sampling_params'] | sampler_params

for item in np.setdiff1d(list(params['sampling_params']), ['distributions', 'state']):
    print("{}: {}".format(item, params['sampling_params'][item]))

state = params['sampling_params']['state']
specs = params['sampling_params']['species']
nagents = len(state)

xt = [state[ag][0] for ag in range(nagents)]
yt = [state[ag][1] for ag in range(nagents)]

grd_truth = {}
for ag in range(nagents):
    gt = get_nhbrs(ag, np.array(xt), np.array(yt), params['sampling_params']['int_rad'])
    grd_truth[ag] = np.setdiff1d(gt, ag)

print("Calculated network: \t\t Ground truth:")
for ag in range(nagents):
    print("{} \t\t {}".format('???', grd_truth[ag]))

# lcn = hiton(0, params['sampling_params']['state'], params)
nwk = hiton_iterator(params['sampling_params']['state'], params)

print("Calculated network: \t\t Ground truth:")
for ag in range(nagents):
    print("{} \t\t {}".format(nwk['tpc'][ag], grd_truth[ag]))

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


