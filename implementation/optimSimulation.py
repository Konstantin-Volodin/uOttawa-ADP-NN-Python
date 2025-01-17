#%%
#region Load Modules
from random import random
from tracemalloc import start
from gurobipy import *
import itertools
import os.path
import os
import json
import pickle
from copy import deepcopy
from multiprocessing import Pool
from functools import partial
from mpi4py import MPI
import time
import sys

import tqdm
from tqdm.keras import TqdmCallback
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
#endregion

#region Simulation
def calculateCost(state, action):
    '''
    ### Input: 2 dictionaries containing current space and a corresponding action
    ### Output: float defining cost
    '''
    cost = 0

    # Divert Cost
    for p in P:
        cost += oc[p] * action['z'][p]

    # Scheduling Cost
    for p,n in itertools.product(P, N):
        cost += book[(p,n)] * action['a'][p][n]

    # Cost of Delay
    for p in P:
        cost += lb[p] * (state['y'][p] - sum(action['a'][p]) - action['z'][p])

    return cost
def ExploitPolicy(state):
    '''
    ### Input: Dictionary containing current space: {"x": x, "y": y}
    ### Output: Dictionary containing action: {"a": a, "z": z}
    ### Policy: Policy uses Myopic cost and also an approximation of cost-to-go function. Approximation is done through a regression model (of different complexities)
    '''
    # Fix State Variables
    for n in N: 
        sx[n].LB = round(state['x'][n])
        sx[n].UB = round(state['x'][n])
    for p in P:
        sy[p].LB = round(state['y'][p])
        sy[p].UB = round(state['y'][p])

    # Optimize
    model.setObjective(exploit_objective, GRB.MINIMIZE)
    model.optimize()

    # If unable to optimize - save the model
    try:
        ac_a = {p:[aa[(p,n)].X for n in N] for p in P}
        ac_z = {p:az[p].X for p in P}
    except:
        print('failed')
        model.params.LogToConsole = 1
        model.optimize()
        model.write(f'nn_{np.random.random()}.lp')

    # Generate Action
    ac_a = {p:[aa[(p,n)].X for n in N] for p in P}
    ac_z = {p:az[p].X for p in P}

    action = {"a": ac_a, "z": ac_z}

    # Return action
    return(action)
def ExplorePolicy(state):
    '''
    ### Input: Dictionary containing current space: {"x": x, "y": y} & weights from the neural network
    ### Output: Dictionary containing action: {"a": a, "z": z}
    ### Policy: First feasible policy that ignores any costs
    '''
    # Fix State Variables
    for n in N: 
        sx[n].LB = round(state['x'][n])
        sx[n].UB = round(state['x'][n])
    for p in P:
        sy[p].LB = round(state['y'][p])
        sy[p].UB = round(state['y'][p])

    # Optimize
    model.setObjective(explore_objective, GRB.MINIMIZE)
    model.optimize()

    # If unable to optimize - save the model
    try:
        ac_a = {p:[aa[(p,n)].X for n in N] for p in P}
        ac_z = {p:az[p].X for p in P}
    except:
        print('failed')
        model.params.LogToConsole = 1
        model.optimize()
        model.write(f'nn_{np.random.random()}.lp')

    # Generate Action
    ac_a = {p:[aa[(p,n)].X for n in N] for p in P}
    ac_z = {p:az[p].X for p in P}

    action = {"a": ac_a, "z": ac_z}

    # Return action
    return(action)
def simulation(state_i, repl, warmup, duration):
    '''
    # Description:
        Estimates a long term discounted cost & average cost for a single input state
    '''
    # Initialize state data
    state_to_evaluate = None
    final_disc_cost = []
    final_avg_cost = []

    random_stream = np.random.RandomState(seed = state_i)
    st_x = [min(random_stream.poisson(cap* 0.95**(n)),cap) for n in N]
    st_y = {p:random_stream.poisson(dm[p]) for p in P}
    state = {'x': st_x, 'y': st_y}

    # Warm up
    for day in range(warmup):

        # generate action
        if random_stream.random() <= epsilon:
            action = ExploitPolicy(state)
        else:
            action = ExplorePolicy(state)

        # action = ExplorePolicy(state)

        # Execute Action
        for p in P:
            state['y'][p] -= action['z'][p]
        for p,n in itertools.product(P, N):
            state['y'][p] -= action['a'][p][n]
            state['x'][n] += action['a'][p][n]

        # Transition
        for p in P:
            state['y'][p] += random_stream.poisson(dm[p])
        for n in N:
            if n != N[-1]: state['x'][n] = state['x'][n+1]
            else: state['x'][n] = 0

    state_to_evaluate = deepcopy(state)

    # Evaluation
    for rep in range(repl):

        disc_cost = 0
        avg_cost = 0
        avg_iter = 0
        state = deepcopy(state_to_evaluate)

        # Single Replication
        for day in range(duration-warmup):
                
            # generate action            
            if random_stream.random() <= epsilon:
                action = ExploitPolicy(state)
            else:
                action = ExplorePolicy(state)
            # action = ExplorePolicy(state)

            # Compute Cost
            cost = calculateCost(state, action)
            disc_cost += cost * (gam**(day-warmup))
            avg_cost += cost
            avg_iter += 1

            # Execute Action
            for p in P:
                state['y'][p] -= action['z'][p]
            for p,n in itertools.product(P, N):
                state['y'][p] -= action['a'][p][n]
                state['x'][n] += action['a'][p][n]

            # Transition
            for p in P:
                state['y'][p] += random_stream.poisson(dm[p])
            for n in N:
                if n != N[-1]: state['x'][n] = state['x'][n+1]
                else: state['x'][n] = 0

        final_disc_cost.append(disc_cost)
        final_avg_cost.append(avg_cost/avg_iter)
    
    return(state_to_evaluate, np.average(final_disc_cost), np.average(final_avg_cost))
#endregion

#region Optimization Functions
def valueApprox(states_range, repl, warmup, duration):
    '''
    # Inputs:
        n_states (int): number of states to generate a value approximation for
        repl (int): number of replications to average over for a single state-approximation
        warmup (int): number of days to stabilize the state for
        duration (int): total number of days a signle replication should run for
    # Output:
        dataframe, with a shape of (n_states, state_parameters + 2). For each state (defined by state variables) there is a overall discounted cost and an average cost generated.
        this dataframe can be used to fit a neural network to modify the policy 
    '''
    states = []
    disc_costs = []
    avg_costs = []

    start_time = time.time()
    for state_iters in states_range:
        state, disc_cost, avg_cost = simulation(state_i = state_iters, repl=repl, warmup=warmup, duration=duration)
        states.append(state)
        disc_costs.append(disc_cost)
        avg_costs.append(avg_cost)
        sys.stdout.flush()
        # print(f'Process {rank} of {size} finished simulations {state_iters}')
    end_time = time.time()
    print(f'Process {rank} of {size} finished simulations {states_range} in {round(end_time-start_time,2)} seconds for {repl}_{warmup}_{duration}')

    # Convert States to Dataframe
    df = pd.DataFrame()
    for p in P: df[f"y_{p}"] = 0
    for n in N: df[f"x_{n+1}"] = 0
    df["disc_cost"] = 0
    df["avg_cost"] = 0

    for i in range(len(states)):
        y = {f"y_{p}":[states[i]['y'][p]] for p in P}
        x = {f"x_{n+1}":[states[i]['x'][n]] for n in N}
        disc_cost = {"disc_cost": [disc_costs[i]]}
        avg_cost = {"avg_cost": [avg_costs[i]]}
        t_df = pd.DataFrame.from_dict({**x, **y, **disc_cost, **avg_cost})
        df = pd.concat([df, t_df])

    return(df)
def fitNN(value_df):
    '''
    # Inputs:
        value_df (dataframe) - dataframe generated thorugh the simulation process above
    # Output:
        neural network weights: outputs a dictionary containing information on weights and biases for each layer within a dense neural network
    '''

    # Initialize Data
    x_cols = value_df.drop(['disc_cost','avg_cost'], axis=1)
    y_cols = value_df['disc_cost']

    # Fit Linear Model
    reg = linear_model.Ridge(alpha=.5)
    reg.fit(x_cols, y_cols)


    # # Split into train and test
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(x_cols, y_cols, test_size = 0.2, train_size = 0.8)

    # # Create the model
    # model = Sequential()
    # for layer in range(len(nn_layers) - 1):
    #     model.add(Dense(nn_layers[layer+1], input_dim=nn_layers[layer], activation="relu", name=f'layer_{layer}'))
    #     if layer != len(nn_layers)-2: model.add(Dropout(rate = 0.01))

    # # Fit the model
    # model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['MeanAbsolutePercentageError'])
    # model.fit(x=x_train, y=y_train, epochs=5000, validation_data=(x_test, y_test), verbose=0, callbacks=[TqdmCallback(verbose=1)])   

    # # Saves weights
    # weights = {}
    # for layer in range(len(nn_layers)-1):
    #     weights[f"layer_{layer}"] = {} 
    #     weights[f"layer_{layer}"]['bias'] = model.layers[layer*2].get_weights()[1].tolist()
    #     weights[f"layer_{layer}"]['weights'] = model.layers[layer*2].get_weights()[0].tolist()

    # # Return weigts
    # return(weights)
    
    return(reg.coef_.tolist())
#endregion

# Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#%%
#region Prepare Data
# Load Data
if rank == 0:
    my_path = os.getcwd()
    data_f = open(os.path.join(my_path, 'data','simple.json'))
    data = json.load(data_f)
    data_f.close()

    # prepare data
    N = [i for i in range(int(data['index']['n']))]
    P = data['index']['p']

    tg = {p:int(data['target'][p]) for p in P}
    dm = {p:int(data['demand'][p]) for p in P}
    cap = int(data['capacity'])
    scap = int(data['surge_capacity'])
    oc = {p:int(data['params']['oc'][p]) for p in P}
    lb = {p:int(data['params']['lb'][p]) for p in P}
    gam = float(data['params']['disc'])

    # booking cost
    book = {(p,n):0 for p in P for n in N}
    for p,n in itertools.product(P, N):
        if (n+1) > tg[p]:
            for k in range(0, (n+1) - tg[p]):
                book[(p,n)] += (gam**k) * lb[p]

    # Neural Network Global Weights
    nn_layers = (len(N)+len(P)+2,     40,40,   1)
    nn_weights = {}
    for layer in range(len(nn_layers)-1):
        ly = f"layer_{layer}"
        nn_weights[ly] ={ "bias": [], "weights": [] }

        for out in range(nn_layers[layer+1]):
            nn_weights[ly]["bias"].append(0)
            
        for inn in range(nn_layers[layer]):
            nn_weights[ly]['weights'].append([])
            for out in range(nn_layers[layer+1]):
                nn_weights[ly]["weights"][inn].append(0)

    # Linear Regression Global Weights
    reg_layers = (len(N)+len(P)+2)
    reg_weights = [0 for it in range(reg_layers)]
else:
    # prepare data
    N = None
    P = None

    tg = None
    dm = None
    cap = None
    scap = None
    oc = None
    lb = None
    gam = None

    # booking cost
    book = None

    # Neural Network Data
    nn_layers = None
    nn_weights = None

    # Linear Regression Weights
    reg_layers = None
    reg_weights = None

# Broadcast Data
N = comm.bcast(N, root=0)
P = comm.bcast(P, root=0)

tg = comm.bcast(tg, root=0)
dm = comm.bcast(dm, root=0)
cap = comm.bcast(cap, root=0)
scap = comm.bcast(scap, root=0)
oc = comm.bcast(oc, root=0)
lb = comm.bcast(lb, root=0)
gam = comm.bcast(gam, root=0)

book = comm.bcast(book, root=0)

nn_layers = comm.bcast(nn_layers, root=0)
nn_weights = comm.bcast(nn_weights, root=0)

reg_layers = comm.bcast(reg_layers, root=0)
reg_weights = comm.bcast(reg_weights, root=0)
#endregion

# Performs Optimization
n_states = 256 * 8 * 3
repl = 100
warmup = 50
duration = 450
alpha = 1
epsilon = 0.9

#region Splits up iterations between each CPU
if rank == 0:
    # Splits up the Data into Sections
    ave, res = divmod(n_states, size)
    counts = [ave + 1 if p < res else ave for p in range(size)]

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(size)]
    ends = [sum(counts[:p+1]) for p in range(size)]

    # converts data into a list of arrays 
    iterable = [range(starts[p],ends[p]) for p in range(size)]
else:
    iterable = None
iterable = comm.scatter(iterable, root=0)
#endregion
# print(f'Process {rank} of {size} received {iterable} for {repl}_{warmup}_{duration}')
sys.stdout.flush()

# Optimization
#%%
for iteri in range(50):

    #region Creates an optimization model
    model = Model()
    model.Params.LogToConsole = 0

    # Variables
    sx = model.addVars(N, vtype=GRB.INTEGER, lb=0, ub=cap, name='sx')
    sy = model.addVars(P, vtype=GRB.INTEGER, lb=0, ub=15, name='sy')
    aa = model.addVars(P, N, vtype=GRB.INTEGER, lb=0, name='aa')
    az = model.addVars(P, vtype=GRB.INTEGER, lb=0, name='az')

    # Post Decision State
    sx_p = model.addVars(N, vtype=GRB.INTEGER, name='sxp')
    sy_p = model.addVars(P, vtype=GRB.INTEGER, name='syp')
    sx_p_tot = model.addVar(vtype=GRB.INTEGER, name='sxp_t')
    sy_p_tot = model.addVar(vtype=GRB.INTEGER, name='syp_t')

    # Linear Regression
    pred_val = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'cost-to-go')

    # # Neutrons & relu
    # vnn = []
    # vrl = []
    # for layer in range(len(nn_layers)-1):
    #     vnn.append(model.addVars(nn_layers[layer+1], lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f'layer{layer}_neuron'))
    #     vrl.append(model.addVars(nn_layers[layer+1], name=f'layer{layer}_RELU'))

    # State Action Constraints
    model.addConstrs( (sx[n] + quicksum(aa[(p,n)] for p in P) <= cap for n in N), name='c_cap')
    model.addConstr( quicksum(az[p] for p in P) <= scap, name='c_scap')
    model.addConstrs( (quicksum(aa[(p,n)] for n in N) + az[p] <= sy[p] for p in P), name='c_dem' )

    # Post Decision State Definition
    model.addConstrs( (sx_p[n] == sx[n] + quicksum( aa[(p,n)] for p in P ) for n in N), name='c_sxp')
    model.addConstrs( ( sy_p[p] == sy[p] - quicksum( aa[(p,n)] for n in N ) - az[p] for p in P ), name='c_syp' )
    model.addConstr( (sx_p_tot == quicksum (sx_p[n] for n in N) ), name='c_sxp_t' )
    model.addConstr( (sy_p_tot == quicksum (sy_p[p] for p in P) ), name='c_syp_t' )

    # # RELU Function Definition
    # for layer in range(len(nn_layers)-1):
    #     for nn_i in range(nn_layers[layer+1]):
    #         model.addConstr( vrl[layer][nn_i] == max_(vnn[layer][nn_i], constant=0), name=f'c_layer{layer}_RELU[{nn_i}]')

    # # Neuron Definition
    # for layer in range(len(nn_layers)-1):
    #     if layer == 0: 
    #         model.addConstrs((
    #             vnn[layer][out] == (
    #                 nn_weights[f"layer_{layer}"]['bias'][out] +
    #                 quicksum( nn_weights[f"layer_{layer}"]["weights"][P.index(p)][out] * sy_p[p] for p in P ) +
    #                 quicksum( nn_weights[f"layer_{layer}"]["weights"][n + len(P)][out] * sx_p[n] for n in N ) +
    #                 ( nn_weights[f"layer_{layer}"]["weights"][-2][out] * sx_p_tot ) + 
    #                 ( nn_weights[f"layer_{layer}"]["weights"][-1][out] * sy_p_tot )
    #             ) for out in range(nn_layers[layer+1])
    #         ), name=f'c_layer{layer}_neuron')
    #     else:
    #         model.addConstrs((
    #             vnn[layer][out] == (
    #                 nn_weights[f"layer_{layer}"]['bias'][out] +
    #                 quicksum( nn_weights[f"layer_{layer}"]["weights"][inp][out] * vrl[layer-1][inp] for inp in range(nn_layers[layer]) ) 
    #             ) for out in range(nn_layers[layer+1])
    #         ), name=f'c_layer{layer}_neuron')

    # Linear Regression Definition
    model.addConstr(
        pred_val == (
            quicksum( reg_weights[P.index(p)] * sy_p[p] for p in P ) +
            quicksum( reg_weights[n + len(P)] * sx_p[n] for n in N ) +
            ( reg_weights[-2] * sx_p_tot ) + 
            ( reg_weights[-1] * sy_p_tot )
        ), name='def-cost-to-go'
    )

    # Objective Function
    exploit_objective = LinExpr(
        quicksum(quicksum( aa[(p,n)] * book[(p,n)] for p in P) for n in N) + 
        quicksum( az[p] * oc[p] for p in P) +
        quicksum( (sy[p] - quicksum(aa[(p,n)] for n in N) - az[p]) * lb[p] for p in P) + 
        # gam * vrl[-1][0]
        gam * pred_val
    )
    explore_objective = LinExpr( 0 )
    #endregion

    # Performs Simulation
    val_portion = valueApprox(iterable, repl, warmup, duration)
    val_portion = comm.gather(val_portion, root=0)

    # Cleans up Simulation Data
    if rank == 0:
        value_data = pd.DataFrame()
        for item in val_portion:
            value_data = pd.concat([value_data, item])
        value_data.to_csv(f'data/sim-optim-logs/simulation-value-iter{iteri}_{repl}_{warmup}_{duration}.csv', index=False)
        if rank == 0: print(value_data['avg_cost'].mean())
    comm.Barrier()

    # Fits a Predictive Model
    # strategy = tf.distribute.MirroredStrategy()

    # Fits a Predictive Model
    if rank == 0:
        value_data['x'] = 0
        for n in N: value_data['x'] += value_data[f'x_{n+1}'] 
        value_data['y'] = 0
        for p in P: value_data['y'] += value_data[f'y_{p}']

        new_reg_weights = fitNN(value_data)

        for betaval in range(len(reg_weights)):
            reg_weights[betaval] = reg_weights[betaval] * (1-alpha) + new_reg_weights[betaval] * (alpha)
    
        # Save Data
        with open(f'data/sim-optim-logs/simulation-betas-iter{iteri}_{repl}_{warmup}_{duration}.pickle', 'wb') as file:
            pickle.dump(reg_weights, file) 
        # print(reg_weights)
    reg_weights = comm.bcast(reg_weights, root=0)
    comm.Barrier()
# %% Visualize Policy

# Import
with open('data/models/sim-optim.pickle', 'rb') as file:
    sim_weights = pickle.load(file) 
y_weights = sim_weights[0:3]
x_weights = sim_weights[3:-2]
y_tot = sim_weights[-1]
x_tot = sim_weights[-2]

# Save in a good format
weights = {'x': {}, 'y': {}}
for n in N:
    weights['x'][n] = x_weights[n] + x_tot

for p in range(len(P)):
    weights['y'][P[p]] = y_weights[p] + y_tot

# Calculate
a_coef = {(p,n):0 for p in P for n in N}
for n in N:
    for p in P:
        a_coef[(p,n)] += book[(p,n)]
        if n != N[0]: a_coef[(p,n)] += gam * weights['x'][n-1]
        a_coef[(p,n)] -= lb[p]
        a_coef[(p,n)] -= gam * weights['y'][p]
coef_df = pd.Series(a_coef).reset_index()
coef_df.columns = ['P','N','Val']
fig = px.line(coef_df, x='N',y='Val',color='P', title='Simulation Approximation - Scheduling Objective', markers=True)
fig.show(renderer="browser")
# %%
