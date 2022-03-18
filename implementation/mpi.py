# from mpi4py import MPI
import ossaudiodev
from re import L
import time
import tqdm
import os
from multiprocessing import Pool

def test(iter):
    print(f'Start {iter} at {time.time()}')
    time.sleep(1)
    print(f'End {iter} at {time.time()}')

if __name__ == "__main__":
    
    for i in range(10):
        print(f'This loop should only be done once {i}')

    n_iters = range(128*2)
    pool_size = os.environ.get('SLURM_NTASKS')
    print(pool_size)
    pool = Pool(pool_size)
    for i in tqdm.tqdm(pool.imap_unordered(test, n_iters), total=len(n_iters)):
        pass
    pool.close()
    
    for i in range(10):
        print(f'This loop should only be done once {i}')