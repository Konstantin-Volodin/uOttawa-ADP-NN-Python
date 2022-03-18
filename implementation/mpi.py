# %%
from tkinter.messagebox import NO
from mpi4py import MPI
import numpy as np
import pandas as pd
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def simulation(iterable):
    state_data = {'state': []}
    cost_data = {'cost': []}
    for iter in iterable:
        time.sleep(1)
        # state data and cost data are generated as part of an expensive function - iter in iterable determines random stream to ensure we are evaluating different things
        # This function uses data that is broadcasted
        state_data['state'].append(iter*i)
        cost_data['cost'].append(iter*i)
        print(f"Process {rank} of {size} completed simulation {iter} ")
    
    df_data = pd.DataFrame.from_dict({**state_data, **cost_data})
    return(df_data)

# Prepare and broadcast data
if rank == 0:
    example_data_1 = np.random.random()
    example_data_2 = np.random.random()
else:
    example_data_1 = None
    example_data_2 = None
example_data_1 = comm.bcast(example_data_1, root=0)
example_data_2 = comm.bcast(example_data_2, root=0)

# Splits iterable portion into chunks
for i in range(3):
    print(f'params {i}')
    iterables = 10
    if rank == 0:
        # Splits up the Data into Sections
        ave, res = divmod(iterables, size)
        counts = [ave + 1 if p < res else ave for p in range(size)]

        # determine the starting and ending indices of each sub-task
        starts = [sum(counts[:p]) for p in range(size)]
        ends = [sum(counts[:p+1]) for p in range(size)]

        # converts data into a list of arrays 
        iterable = [range(starts[p],ends[p]) for p in range(size)]
    else:
        iterable = None
    iterable = comm.scatter(iterable, root=0)
    print(f'Process {rank} of {size} received {iterable} for ')

    sim_data_portion = simulation(iterable)
    sim_data_portion = comm.gather(sim_data_portion, root=0)
    
    if rank == 0:
        sim_data = pd.DataFrame()
        for item in sim_data_portion:
            sim_data = pd.concat([sim_data, item])
        print(sim_data)
    comm.Barrier()

# %%
