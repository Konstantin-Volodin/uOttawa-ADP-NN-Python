# %%
from tkinter.messagebox import NO
from mpi4py import MPI
import numpy as np
import pandas as pd
import time
import tensorflow as tf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
