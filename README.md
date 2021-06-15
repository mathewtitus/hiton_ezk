# README.md

This package is designed to infer the causal structure of a complex simulated environment utilizing the HITON-EZK algorithm.

Parameters for executing the algorithm (number N, N1, N2 of samples drawn at various steps; threshold for on c_ij^K for causal dependence, etc.) are stored in a dictionary and passed into the `hiton` or `hiton_iterator` functions as the `params` keyword argument. This also contains a dictionary under the key `sampler_params` which holds the information needed to execute a simulation, or perturb a system state, and will depend on the simulation function (stored in params['sampling_params']['simulator']) the user provides.

The simulation of interest must be packaged in a function `simulator` so that we can generate simulation output by a call of the form 
`simulator(**kwargs)`
with kwargs a dictionary of inputs. The simulator function is assigned to the key value `simulator` in params['sampler_params']

In this version the package only supports single target variable testing, so questions of whether agent j is influencing agent i must be expressed as ``Does the collection of variables describing agent j have a statistical effect on the single variable x of agent i?'' The choice of variable x is made in params['sampling_params']['target_variable']. Of course, in some applications the response variable may not be one-dimensional; in a future update this will be supported, allowing us to rephrase the above question as ``Does the collection of agent j's variables have a statistical influence on any one of the variables defining agent i?''
