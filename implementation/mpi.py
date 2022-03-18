# from mpi4py import MPI
import time
import tqdm
from multiprocessing import Pool

def test(iter):
    print(f'Start {iter} at {time.time()}')
    time.sleep(1)
    print(f'End {iter} at {time.time()}')

if __name__ == "__main__":
    
    n_iters = range(128*2)
    pool = Pool(128)
    for i in tqdm.tqdm(pool.imap_unordered(test, n_iters), total=len(n_iters)):
        pass
    pool.close()


    # world_comm = MPI.COMM_WORLD
    # world_size = world_comm.Get_size()
    # my_rank = world_comm.Get_rank()

    

    # print(world_comm)
    # print("World Size: " + str(world_size) + "   " + "Rank: " + str(my_rank))