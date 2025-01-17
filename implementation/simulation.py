#%% 
#region Load Modules
from copyreg import pickle
from gurobipy import *
import itertools
import os.path
import json
import pickle
from copy import deepcopy
from multiprocessing import Pool
from functools import partial
from stable_baselines3 import PPO

from optimRL import SchedEnv

from tqdm import trange, tqdm
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
#endregion
#region Load & Prepare Data
# load data
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

# expected values
E_x = [ (cap *(0.95**i)) for i in N]
E_y = {p:dm[p] for p in P}

# booking cost
book = {(p,n):0 for p in P for n in N}
for p,n in itertools.product(P, N):
    if (n+1) > tg[p]:
        for k in range(0, (n+1) - tg[p]):
            book[(p,n)] += (gam**k) * lb[p]


# Sim Weights
#endregion

#region Simulation Functions
def calculateCost(state, action):
    '''
    ### Input: 2 dictionaries containing current space and a corresponding action
    ### Output: float defining cost
    '''

    state = deepcopy(state)
    st_x = state['x']
    st_y = state['y']

    action = deepcopy(action)
    ac_a = action['a']
    ac_z = action['z']

    cost = 0

    # Divert Cost
    for p in P:
        cost += oc[p] * ac_z[p]

    # Scheduling Cost
    for p,n in itertools.product(P, N):
        cost += book[(p,n)] * ac_a[p][n]

    # Cost of Delay
    for p in P:
        cost += lb[p] * (st_y[p] - sum(ac_a[p]) - ac_z[p])

    return cost
def MyopicPolicy(state):
    '''
    ### Input: Dictionary containing current space: {"x": x, "y": y}
    ### Output: Dictionary containing action: {"a": a, "z": z}
    ### Policy: Policy that schedules into first available slot (and uses overtime asap) in order of decreasing priority
    '''
    # Initialize 
    state = deepcopy(state)
    myopic = Model('PPQ')
    myopic.Params.LogToConsole = 0
    
    # Variables
    sx = myopic.addVars(N, vtype=GRB.INTEGER, lb=0, ub=cap, name='sx')
    sy = myopic.addVars(P, vtype=GRB.INTEGER, lb=0, ub=15, name='sy')
    aa = myopic.addVars(P, N, vtype=GRB.INTEGER, lb=0, name='aa')
    az = myopic.addVars(P, vtype=GRB.INTEGER, lb=0, name='az')

    # State Action Constraints
    ccap = myopic.addConstrs( (sx[n] + quicksum(aa[(p,n)] for p in P) <= cap for n in N), name='sc_cap')
    cscap = myopic.addConstr( quicksum(az[p] for p in P) <= scap, name='sc_scap')
    cdem = myopic.addConstrs( (quicksum(aa[(p,n)] for n in N) + az[p] <= sy[p] for p in P), name='sc_dem' )

    # Fix State Variables
    for n in N: 
        sx[n].LB = state['x'][n]
        sx[n].UB = state['x'][n]
    for p in P:
        sy[p].LB = state['y'][p]
        sy[p].UB = state['y'][p]

    # Objective Function
    myopic.setObjective((
        quicksum(quicksum( aa[(p,n)] * book[(p,n)] for p in P) for n in N) + 
        quicksum( az[p] * oc[p] for p in P) +
        quicksum( (sy[p] - quicksum(aa[(p,n)] for n in N) - az[p]) * lb[p] for p in P)
    ), GRB.MINIMIZE)

    # Optimize
    myopic.optimize()
    
    # Generate Action
    ac_a = {p:[aa[(p,n)].X for n in N] for p in P}
    ac_z = {p:az[p].X for p in P}
    action = {"a": ac_a, "z": ac_z}

    # Return action
    return(action)
def DMBPolicy(state, betas):
    pass
def PPQPolicy(state, **kwargs):
    '''
    ### Input: 
    #       Dictionary containing current space: {"x": x, "y": y}
    #       Dictionary containing Beta values that define an optimal policy
    ### Output: Dictionary containing action: {"a": a, "z": z}
    ### Policy: Policy is based on Jonathan et al (2008)
    '''
    # Initialize 
    state = deepcopy(state)
    betas=kwargs['kwargs']['betas']
    ppq = Model('PPQ')
    ppq.Params.LogToConsole = 0
    
    # Variables
    sx = ppq.addVars(N, vtype=GRB.INTEGER, lb=0, ub=cap, name='sx')
    sy = ppq.addVars(P, vtype=GRB.INTEGER, lb=0, ub=15, name='sy')
    aa = ppq.addVars(P, N, vtype=GRB.INTEGER, lb=0, name='aa')
    az = ppq.addVars(P, vtype=GRB.INTEGER, lb=0, name='az')

    # State Action Constraints
    ccap = ppq.addConstrs( (sx[n] + quicksum(aa[(p,n)] for p in P) <= cap for n in N), name='sc_cap')
    cscap = ppq.addConstr( quicksum(az[p] for p in P) <= scap, name='sc_scap')
    cdem = ppq.addConstrs( (quicksum(aa[(p,n)] for n in N) + az[p] <= sy[p] for p in P), name='sc_dem' )

    # Fix State Variables
    for n in N: 
        sx[n].LB = state['x'][n]
        sx[n].UB = state['x'][n]
    for p in P:
        sy[p].LB = state['y'][p]
        sy[p].UB = state['y'][p]

    # Objective Function
    ppq.setObjective((
        quicksum(quicksum( aa[(p,n)] * (
            book[(p,n)] + gam*betas['bx'][n] - lb[p] - gam*betas['by'][p]
        ) for p in P) for n in N) + 
        quicksum( az[p] * (
            oc[p] - lb[p] - gam*betas['by'][p]
        ) for p in P)
    ), GRB.MINIMIZE)

    # Optimize
    ppq.optimize()
    
    # Generate Action
    ac_a = {p:[aa[(p,n)].X for n in N] for p in P}
    ac_z = {p:az[p].X for p in P}
    action = {"a": ac_a, "z": ac_z}

    # Return action
    return(action)
def RLPolicy(state, **kwargs):
    '''
    ### Input: 
    #       Dictionary containing current space: {"x": x, "y": y}
    #       Model file 
    ### Output: Dictionary containing action: {"a": a, "z": z}
    ### Policy: Policy is created using Sample Baselines 3 with algorithm with custom environment
    '''
    model=kwargs['kwargs']['model']

    # Modifies State
    modified_state = deepcopy(state)
    modified_state['x'] = [modified_state['x']]
    modified_state['y'] = [[modified_state['y'][p] for p in P]]

    # Generate action
    raw_actions = model.predict(modified_state, deterministic=True)
    raw_actions = raw_actions[0].reshape(len(P), len(N) + 1)
    raw_ac_z = np.round(raw_actions[:,0])
    raw_ac_a = np.round(raw_actions[:,1:])

    # Ajust Action
    ac_a = {p:[0 for n in N] for p in P}
    ac_z = {p:0 for p in P}
    action = {"a": ac_a, "z": ac_z}

    # Divert Patients
    rem_cap = scap
    for p in range(len(P)):
        action_temp = min(raw_ac_z[p], modified_state['y'][0][p], rem_cap)
        rem_cap -= action_temp
        modified_state['y'][0][p] -= action_temp
        action['z'][P[p]] = action_temp

    # Schedule remaining patients
    for p in range(len(P)):
        for n in N:
            rem_cap = cap - modified_state['x'][0][n]
            action_temp = min(raw_ac_a[p][n], modified_state['y'][0][p], rem_cap)
            modified_state['x'][0][n] += action_temp
            modified_state['y'][0][p] -= action_temp
            action['a'][P[p]][n] = action_temp
            
    return(action)
def SimPolicy(state, **kwargs):
    '''
    ### Input: 
    #       Dictionary containing current space: {"x": x, "y": y}
    #       Dictionary containing weights corresponding to a solved Neural Network
    ### Output: Dictionary containing action: {"a": a, "z": z}
    ### Policy: Policy is based on Saure et al (2015)
    '''
    '''
    ### Input: Dictionary containing current space: {"x": x, "y": y} & weights from the neural network
    ### Output: Dictionary containing action: {"a": a, "z": z}
    ### Policy: Policy uses Myopic cost and also an approximation of cost-to-go function. Approximation is done through a neural network
    '''
    # Initialize 
    state = deepcopy(state)
    weights=kwargs['kwargs']['weights']
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

    # State Action Constraints
    model.addConstrs( (sx[n] + quicksum(aa[(p,n)] for p in P) <= cap for n in N), name='c_cap')
    model.addConstr( quicksum(az[p] for p in P) <= scap, name='c_scap')
    model.addConstrs( (quicksum(aa[(p,n)] for n in N) + az[p] <= sy[p] for p in P), name='c_dem' )

    # Post Decision State Definition
    model.addConstrs( (sx_p[n] == sx[n] + quicksum( aa[(p,n)] for p in P ) for n in N), name='c_sxp')
    model.addConstrs( ( sy_p[p] == sy[p] - quicksum( aa[(p,n)] for n in N ) - az[p] for p in P ), name='c_syp' )
    model.addConstr( (sx_p_tot == quicksum (sx_p[n] for n in N) ), name='c_sxp_t' )
    model.addConstr( (sy_p_tot == quicksum (sy_p[p] for p in P) ), name='c_syp_t' )

    # Linear Regression Definition
    model.addConstr(
        pred_val == (
            quicksum( weights[P.index(p)] * sy_p[p] for p in P ) +
            quicksum( weights[n + len(P)] * sx_p[n] for n in N ) +
            ( weights[-2] * sx_p_tot ) + 
            ( weights[-1] * sy_p_tot )
        ), name='def-cost-to-go'
    )
    
    # Objective Function
    model.setObjective((
        quicksum(quicksum( aa[(p,n)] * book[(p,n)] for p in P) for n in N) + 
        quicksum( az[p] * oc[p] for p in P) +
        quicksum( (sy[p] - quicksum(aa[(p,n)] for n in N) - az[p]) * lb[p] for p in P) + 
        gam * pred_val
    ), GRB.MINIMIZE)

    # Fix State Variables
    for n in N: 
        sx[n].LB = round(state['x'][n])
        sx[n].UB = round(state['x'][n])
    for p in P:
        sy[p].LB = round(state['y'][p])
        sy[p].UB = round(state['y'][p])

    # Optimize
    model.optimize()

    # Generate Action
    ac_a = {p:[aa[(p,n)].X for n in N] for p in P}
    ac_z = {p:az[p].X for p in P}
    action = {"a": ac_a, "z": ac_z}

    # Return action
    return(action)

def simulation(repl_i, warmup=1000, duration=2000, debug=False, policy=MyopicPolicy, **kwargs):
    random_stream = np.random.RandomState(seed = repl_i)
    st_x = [min(random_stream.poisson(cap*(0.9)**n), cap) for n in N]
    st_y = {p:min(random_stream.poisson(dm[p]),dm[p]*5) for p in P}
    state = {'x': st_x, 'y': st_y}
    disc_cost = 0
    avg_cost = 0
    avg_iter = 0

    # Single Replication
    for day in range(duration):

        # generate action
        if day < warmup:
            action = MyopicPolicy(state)
        else:   
            if kwargs: action = policy(state, kwargs = kwargs['kwargs'])
            else: action = policy(state)
        
        # generate cost
        if day >= warmup:
            cost = calculateCost(state, action)
            disc_cost = (disc_cost * gam) + cost
            avg_cost += cost
            avg_iter += 1

        if debug:
            print(f"\tDay {day+1} cost: {cost}, disc cost: {disc_cost}")

        # Execute Action
        for p in P:
            st_y[p] -= action['z'][p]
        for p,n in itertools.product(P, N):
            st_y[p] -= action['a'][p][n]
            st_x[n] += action['a'][p][n]

        # Transition
        for p in P:
            st_y[p] += random_stream.poisson(dm[p])
        for n in N:
            if n != N[-1]: st_x[n] = st_x[n+1]
            else: st_x[n] = 0

    return(disc_cost, avg_cost/avg_iter)
#endregion

#region Simulation
replications = [i for i in range(100)]

fas_costs_dc = []
fas_costs_avg = []

# PPQ Betas
with open('data/models/ppq.pickle', 'rb') as file:
    PPQbetas = pickle.load(file)
ppq_costs_dc = []
ppq_costs_avg = []

# Sim-Optim More Repls Weights
with open('data/models/sim-optim.pickle', 'rb') as file:
    sim_weights_mr = pickle.load(file) 
sim_costs_dc_mr = []
sim_costs_avg_mr = []

# Sim-Optim Less Repls Weights
with open('data/models/simulation-betas-iter42_10_50_450.pickle', 'rb') as file:
    sim_weights_lr = pickle.load(file) 
sim_costs_dc_lr = []
sim_costs_avg_lr = []

# Sim-Optim Med Repls Weights
with open('data/models/simulation-betas-iter5_100_50_450.pickle', 'rb') as file:
    sim_weights_ar = pickle.load(file) 
sim_costs_dc_ar = []
sim_costs_avg_ar = []


# RL Model
rl_env = SchedEnv(N, P, cap, scap, dm, tg, oc, lb, gam)
rl_model = PPO.load("data/models/rl-ppo", env=rl_env)
rl_costs_dc = []
rl_costs_avg = []

#%%
if __name__ == '__main__':
    pool = Pool(os.cpu_count())
    # FAS
    for disc_cost, avg_cost in tqdm(pool.imap_unordered(partial(simulation, policy=MyopicPolicy), replications), total=len(replications)):
        fas_costs_dc.append(disc_cost)
        fas_costs_avg.append(avg_cost)
    # PPQ
    for disc_cost, avg_cost in tqdm(pool.imap_unordered(partial(simulation, policy=PPQPolicy, kwargs={'betas':PPQbetas}), replications), total=len(replications)):
        ppq_costs_dc.append(disc_cost)
        ppq_costs_avg.append(avg_cost)
    # SIM-Optim MR
    for disc_cost, avg_cost in tqdm(pool.imap_unordered(partial(simulation, policy=SimPolicy, kwargs={'weights':sim_weights_mr}), replications), total=len(replications)):
        sim_costs_dc_mr.append(disc_cost)
        sim_costs_avg_mr.append(avg_cost)
    # SIM-Optim ar  
    for disc_cost, avg_cost in tqdm(pool.imap_unordered(partial(simulation, policy=SimPolicy, kwargs={'weights':sim_weights_ar}), replications), total=len(replications)):
        sim_costs_dc_ar.append(disc_cost)
        sim_costs_avg_ar.append(avg_cost)
    # SIM-Optim LR  
    for disc_cost, avg_cost in tqdm(pool.imap_unordered(partial(simulation, policy=SimPolicy, kwargs={'weights':sim_weights_lr}), replications), total=len(replications)):
        sim_costs_dc_lr.append(disc_cost)
        sim_costs_avg_lr.append(avg_cost)
    # RL
    for repl in tqdm(replications):
        disc_cost, avg_cost = simulation(repl, policy=RLPolicy, kwargs={'model':rl_model})
        rl_costs_dc.append(disc_cost)
        rl_costs_avg.append(avg_cost)
    pool.terminate()

    print(f"Myopic Costs: mean: {round(np.mean(fas_costs_avg),2)} disc: {round(np.mean(fas_costs_dc),2)}")
    print(f"PPQ Costs: mean: {round(np.mean(ppq_costs_avg),2)} disc: {round(np.mean(ppq_costs_dc),2)}")
    print(f"SimOptim Costs More Repls: mean: {round(np.mean(sim_costs_avg_mr),2)} disc: {round(np.mean(sim_costs_dc_mr),2)}")
    print(f"SimOptim Costs Less Repls: mean: {round(np.mean(sim_costs_avg_lr),2)} disc: {round(np.mean(sim_costs_dc_lr),2)}")
    print(f"SimOptim Costs Medium Repls: mean: {round(np.mean(sim_costs_avg_ar),2)} disc: {round(np.mean(sim_costs_dc_ar),2)}")
    print(f"RL Costs: mean: {round(np.mean(rl_costs_avg),2)} disc: {round(np.mean(rl_costs_dc),2)}")
#endregion
# %%