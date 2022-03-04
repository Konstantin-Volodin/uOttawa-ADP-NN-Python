# %%
#region Load Modules
from gurobipy import *
import itertools
import os.path
import pickle
import json
import plotly.express as px
import pandas as pd
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
#endregion

#region Master Model
master = Model("Master")
master.params.LogToConsole = 0

# Goal Variables
mv_b0 = master.addVar(vtype = GRB.CONTINUOUS, lb=0, name='mv_b0')
mv_bx = master.addVars(N, vtype = GRB.CONTINUOUS, lb=0, name='mv_bx')
mv_by = master.addVars(P, vtype = GRB.CONTINUOUS, lb=0, name='mv_by')

# Constraints
mc_b0 = master.addConstr(mv_b0 == 1, 'mc_b0')
mc_bx = master.addConstrs((mv_bx[n] >= E_x[n] for n in N), name='mc_bx')
mc_by = master.addConstrs((mv_by[p] >= E_y[p] for p in P), name='mc_by')

# Objective Function
mo_bx = quicksum(mv_bx[n] for n in N)
mo_by = quicksum(mv_by[p] for p in P)

master.setObjective( (mv_b0 + mo_bx + mo_by), GRB.MINIMIZE)
#endregion

#region Sub Model
sub = Model("Sub")
sub.params.LogToConsole = 0

# Variables
sv_sx = sub.addVars(N, vtype=GRB.INTEGER, lb=0, ub=cap, name='sv_x')
sv_sy = sub.addVars(P, vtype=GRB.INTEGER, lb=0, ub=15, name='sv_y')
sv_aa = sub.addVars(P, N, vtype=GRB.INTEGER, lb=0, name='sv_y')
sv_az = sub.addVars(P, vtype=GRB.INTEGER, lb=0, name='sv_y')

# State Action Constraints
sc_cap = sub.addConstrs( (sv_sx[n] + quicksum(sv_aa[(p,n)] for p in P) <= cap for n in N), name='sc_cap')
sc_scap = sub.addConstr( quicksum(sv_az[p] for p in P) <= scap, name='sc_scap')
sc_dem = sub.addConstrs( (quicksum(sv_aa[(p,n)] for n in N) + sv_az[p] <= sv_sy[p] for p in P), name='sc_dem' )
#endregion

#region Phase 1
iter = 0
mo_p2 = LinExpr()

while True:

    # Solve Master
    master.optimize()
    print(f"PHASE 1 Master Iter {iter}:\t\t{master.ObjVal}")

    # Generate Value Equation
    val_b0 = LinExpr(1-gam)
    val_bx = {}
    for n in N:
        val_bx[n] = LinExpr(sv_sx[n])
        if n != N[-1]: val_bx[n] -= gam * (sv_sx[n+1] + quicksum(sv_aa[(p,n+1)] for p in P))
    val_by = {}
    for p in P: 
        val_by[p] = (1-gam) * sv_sy[p] + gam * (quicksum(sv_aa[p,n] for n in N) + sv_az[p] - E_y[p])
    
    # Update Subproblem
    so_val = ( 
        (mc_b0.Pi * val_b0) + 
        quicksum( mc_bx[n].Pi * val_bx[n] for n in N ) + 
        quicksum( mc_by[p].Pi * val_by[p] for p in P ) 
    )
    so_cost = (
        quicksum( book[(p,n)] * sv_aa[p,n] for p in P for n in N) +
        quicksum( oc[p] * sv_az[p] for p in P ) +
        quicksum( lb[p] * (sv_sy[p] - quicksum( sv_aa[p,n] for n in N ) - sv_az[p]) for p in P)
    )
    sub.setObjective( -so_val, GRB.MINIMIZE )

    # Solve Subproblem
    sub.optimize()
    # print(f"PHASE 1 Sub Iter {iter}:\t\t{sub.ObjVal}")
    
    # Update Master
    sa = Column()
    sa.addTerms(val_b0.getValue(), mc_b0)
    [sa.addTerms(val_bx[n].getValue(), mc_bx[n]) for n in N]
    [sa.addTerms(val_by[p].getValue(), mc_by[p]) for p in P]
    sa_var = master.addVar(vtype = GRB.CONTINUOUS, name= f"sa_{iter}", column = sa)
    
    # Save objective for phase 2
    mo_p2.add(sa_var, so_cost.getValue())
    
    # End Condition
    if master.ObjVal <= 0: 
        master.optimize()
        break

    iter += 1

#endregion

#region Phase 2
master.remove(mv_b0)
master.remove(mv_bx)
master.remove(mv_by)
master.setObjective(mo_p2, GRB.MINIMIZE)

iter += 1
count_same = 0
objs = []
while True:
    
    # Solve Master
    master.optimize()
    # print(f"PHASE 2 Master Iter {iter}:\t\t{master.ObjVal}")

    # Generate Value Equation
    val_b0 = LinExpr(1-gam)
    val_bx = {}
    for n in N:
        val_bx[n] = LinExpr(sv_sx[n])
        if n != N[-1]: val_bx[n] -= gam * (sv_sx[n+1] + quicksum(sv_aa[(p,n+1)] for p in P))
    val_by = {}
    for p in P: 
        val_by[p] = (1-gam) * sv_sy[p] + gam * (quicksum(sv_aa[p,n] for n in N) + sv_az[p] - E_y[p])
    
    # Update Subproblem
    so_val = ( 
        (mc_b0.Pi * val_b0) + 
        quicksum( mc_bx[n].Pi * val_bx[n] for n in N ) + 
        quicksum( mc_by[p].Pi * val_by[p] for p in P ) 
    )
    so_cost = (
        quicksum( book[(p,n)] * sv_aa[p,n] for p in P for n in N) +
        quicksum( oc[p] * sv_az[p] for p in P ) +
        quicksum( lb[p] * (sv_sy[p] - quicksum( sv_aa[p,n] for n in N ) - sv_az[p]) for p in P)
    )
    sub.setObjective( so_cost-so_val, GRB.MINIMIZE )

    # Solve Subproblem
    sub.optimize()
    print(f"PHASE 2 Sub Iter {iter}:\t\t{sub.ObjVal}")
    
    # Update Master
    sa = Column()
    sa.addTerms(val_b0.getValue(), mc_b0)
    [sa.addTerms(val_bx[n].getValue(), mc_bx[n]) for n in N]
    [sa.addTerms(val_by[p].getValue(), mc_by[p]) for p in P]
    sa_var = master.addVar(vtype = GRB.CONTINUOUS, name= f"sa_{iter}", column = sa, obj=so_cost.getValue())
    
    # End Condition
    if sub.ObjVal >= 0:
        master.optimize()
        break
    if count_same >= 100:
        master.optimize()
        break
    
    objs.append(sub.ObjVal)
    objs = objs[-2:]
    if len(objs) >= 2 and objs[-1] == objs[-2]: count_same += 1
    else: count_same = 0

    iter += 1

#endregion

#region Save Data
# Save Model
master.write('data/linear-p2.lp')

# Save Betas
betas = {'b0': mc_b0.Pi, 'bx': {}, 'by': {}}
for n in N: betas['bx'][n] = mc_bx[n].Pi
for p in P: betas['by'][p] = mc_by[p].Pi

with open(os.path.join(my_path, 'data','linear-betas.pkl'), 'wb') as outp:
    pickle.dump(betas, outp, pickle.HIGHEST_PROTOCOL)
#endregion

# %%
#region Factorizing
a_coef = {(p,n):0 for p in P for n in N}
for n in N:
    for p in P:
        a_coef[(p,n)] += book[(p,n)]
        if n != N[0]: a_coef[(p,n)] += gam * betas['bx'][n-1]
        a_coef[(p,n)] -= lb[p]
        a_coef[(p,n)] -= gam * betas['by'][p]
coef_df = pd.Series(a_coef).reset_index()
coef_df.columns = ['P','N','Val']
fig = px.line(coef_df, x='N',y='Val',color='P', title='Linear Approximation - Scheduling Objective', markers=True)
fig.show(renderer="browser")
#endregion
# %%
