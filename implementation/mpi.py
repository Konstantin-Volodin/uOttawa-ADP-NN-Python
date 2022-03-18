# from mpi4py import MPI
import ossaudiodev
from re import L
import time
import tqdm
import os
from multiprocessing import Pool
import numpy as np

def test(iter, rank, size):
    print(f'Start {iter} at {time.time()} on {rank} of {size}')
    time.sleep(1)
    print(f'End {iter} at {time.time()} on {rank} of {size}')

if __name__ == "__main__":
    
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print('Hello from process {} out of {}'.format(rank, size))

    if rank == 0:
        data = np.arange(4.0)
    else:
        data = None

    data = comm.bcast(data, root=0)

    if rank == 0:
        print('Process {} broadcast data:'.format(rank), data)
    else:
        print('Process {} received data:'.format(rank), data)
    
    # for i in range(10):
    #     print(f'This end loop should only be done once {i}')