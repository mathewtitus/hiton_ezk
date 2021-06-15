# data_structures.py
# Mathew Titus, November 2020
# 
# Data classes holding the variables of the simulated 
# system. 
# 
# CpnVar contains the variable's value, topology
# distribution when randomly sampled, and notion of 
# distance.
# 
# CpnSub collects a number of CpnVar objects into an 
# array to allow list operations, as well as a 
# condition method, converting random variables to 
# constant values (a new CpnSub object is returned).
# 
######################################################

import numpy as np
from typing import List

## class holding the variable to be processed in an independence test

# possible distributions are in np.random module (uniform, normal, exponential...)
# with parameters given in dict 
# uniform : low, high
# constant : low, high (equal)
# normal : loc, scale

# topology is a dict with key R1, S1, or T1
# and value NaN or, for T1, L with L the length of torus
# R1 = (-inf, inf)
# S1 = [-pi, pi)
# T1 = [0, L)
class CpnVar:
    # constructor
    def __init__(self, variable, params: dict, topology: dict):
        self.rv = variable
        self.parameters = params
        self.value = np.nan
        self.topology = list(topology)[0]
        self.length = topology[self.topology]
        assert len(topology) == 1

    # define the variable's notion of distance
    def dist(self, val1: float, val2: float):
        if self.topology == "R1":
            return np.abs(val2 - val1)
        elif self.topology == "S1":
            sval1 = np.mod(val1, 2*np.pi); sval2 = np.mod(val2, 2*np.pi);
            ccw_dist = np.abs(sval2 - sval1)
            if ccw_dist < np.pi:
                return ccw_dist
            else:
                return 2 * np.pi - ccw_dist
        elif self.topology == "T1":
            tval1 = np.mod(val1, self.length); tval2 = np.mod(val2, self.length);
            euc_dist = np.abs(tval2 - tval1)
            if euc_dist < self.length / 2:
                return euc_dist
            else:
                return self.length - euc_dist
        else:
            raise Exception

    # convert a real input to an appropriate value in the variable domain
    def project(self, val):
        if self.topology == "R1":
            return val
        elif self.topology == "S1":
            s1_val = np.mod(val, 2*np.pi)
            if s1_val >= np.pi:
                s1_val = s1_val - 2*np.pi
            return s1_val
        elif self.topology == "T1":
            t1_val = np.mod(val, self.length)
            return t1_val
        else:
            raise Exception

    # instantiate the variable with value *val*, perturbation defaults to this value also
    def set_val(self, val: float):
        proj_val = self.project(val)
        self.value = proj_val
        self.perturbation = proj_val

    # replace variable value with an independent draw from its null distribution
    def resample(self):
        # some text to see if it shows up in the documentation
        self.set_val( self.rv.__call__(**self.parameters) )
    
    # give an example value of this variable
    def test_sample(self):
        sample = self.rv.__call__(**self.parameters)
        return self.project(sample)
        
    # based on noise and variable type, perturb CpnVar around its value
    def perturb(self, distribution, epsilon: float = 0.1):
        # check that variable is set
        if np.isnan(self.value): 
            print("Cannot perturb as no value is set for this CPN variable.")
            return False
        # sVample according to given distribution, centered at current value
        if distribution.__name__ == "uniform":
            a = self.value - epsilon / 2
            b = self.value + epsilon / 2
            self.perturbation = distribution.__call__(low=a, high=b)
        elif distribution.__name__ == 'normal':
            m = self.value
            s = epsilon
            self.perturbation = distribution.__call__(loc = m, scale = s)
        elif distribution.__name__ == 'randint':
            self.perturbation = distribution.__call__(low=self.params['low'], high=self.params['high'])
        else:
            print("Not a supported perturbation type ({}).".format(distribution.__name__))

    # copy CpnVar contents to a new object
    def copy(self):
        new_copy = CpnVar(self.rv, self.parameters, {self.topology: self.length})
        new_copy.value = self.value
        try:
            new_copy.perturbation = self.perturbation
        except:
            pass
        return new_copy


# CpnSub objects will hold sets of CpnVar elements
class CpnSub:
    # initialize with list of CpnVar objects
    def __init__(self, Members):
        self.members = Members

    # let len be called on CpnSub
    def __len__(self):
        return len(self.members)

    # let CpnSub class be indexed directly
    def __getitem__(self, index: int):
        return self.members[index]

    # let CpnSub elements be replaced by indexed calls
    def __setitem__(self, index: int, data: CpnVar):
        self.members[index] = data

    def copy(self):
        cpy_members = [cpn_var.copy() for cpn_var in self.members]
        cpn_copy = CpnSub(cpy_members)
        return cpn_copy

    # another way to handle element reassignment
    def replace(self, index: int, var: CpnVar):
        self.members[index] = var

    # extend the member set
    def append(self, var: CpnVar):
        self.members.append(var)

    # remove a member by index
    def pop(self, ind: int):
        assert ind < len(self.members), "Index {} is out of range (CpnSub has {} elements).".format(ind, len(self.members))
        self.members.pop(ind)

    # replace random variables in the sub with constant values
    # NaN's indicate an unconditioned variable
    def condition(self, conds: List, pert_params: dict):
        conditioned_cpn = self.copy()
        to_replace = np.where(~np.isnan(conds))[0]
        for ind in np.arange(len(conds)):
            if ind in to_replace:
                # create constant variable
                const_var = CpnVar(
                                np.random.uniform,                          # variable object
                                {'low': conds[ind], 'high': conds[ind]},    # parameters
                                {"R1": np.nan}                              # topology
                            )
                # initialize the CpnVar object at given value
                const_var.resample()
                # replace the old variable
                conditioned_cpn[ind] = const_var
            else:
                # other variables get perturbed
                # print("not conditioning var {}".format(ind))
                cv = conditioned_cpn[ind]
                # print("it begins with perturbation {}".format(cv.perturbation))
                cv_params = pert_params[ind]
                cv.perturb(cv_params['variable'], cv_params['epsilon'])
                # print("it ends with perturbation {}".format(cv.perturbation))
                # print("our CpnSub object holds the value {}".format(conditioned_cpn[ind].perturbation))
        return conditioned_cpn

    # TODO: create this method to alleviate some of the code in `eliminate` in hiton.py
    def perturb(self):
        pass

    # send variable states to a list of constants
    def values_to_list(self):
        out_list = []
        for m in self.members:
            out_list.append(m.value)
        return out_list

    # send variable states to a list of constants
    def perts_to_list(self):
        out_list = []
        for m in self.members:
            out_list.append(m.perturbation)
        return out_list

