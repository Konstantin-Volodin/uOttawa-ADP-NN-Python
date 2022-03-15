#%% 
from gurobipy import *
import os
from matplotlib import markers

from tqdm.keras import TqdmCallback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json


import matplotlib.pyplot as plt
from sklearn import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras import regularizers

# Load Params
my_path = os.getcwd()
data_f = open(os.path.join(my_path, 'data','simple.json'))
data = json.load(data_f)
data_f.close()

# prepare data
N = [i for i in range(int(data['index']['n']))]
P = data['index']['p']

# Load Data
df_50_0_100 = pd.read_csv('data/simulation-value_50_0_100.csv')
df_5_0_100 = pd.read_csv('data/simulation-value_5_0_100.csv')
df_5_500_600 = pd.read_csv('data/simulation-value_5_500_600.csv')
df_100_0_50 = pd.read_csv('data/simulation-value_100_0_50.csv')
df_100_100_150 = pd.read_csv('data/simulation-value_100_100_150.csv')
df_100_100_200 = pd.read_csv('data/simulation-value_100_100_200.csv')
df_100_100_300 = pd.read_csv('data/simulation-value_100_100_300.csv')
df_100_100_500 = pd.read_csv('data/simulation-value_100_100_500.csv')
df_100_200_250 = pd.read_csv('data/simulation-value_100_200_250.csv')
df_200_100_300 = pd.read_csv('data/simulation-value_200_100_300.csv')

# Modify Data
df_50_0_100['type'] = '50_0_100'
df_5_0_100['type'] = '5_0_100'
df_5_500_600['type'] = '5_500_600'
df_100_0_50['type'] = '100_0_50'
df_100_100_150['type'] = '100_100_150'
df_100_100_200['type'] = '100_100_200'
df_100_100_300['type'] = '100_100_300'
df_100_100_500['type'] = '100_100_500'
df_100_200_250['type'] = '100_200_250'
df_200_100_300['type'] = '200_100_300'

# %% Graph Data
# Review Data
df_tot = pd.concat([df_100_0_50, df_100_100_150, df_100_200_250, df_100_100_200, df_100_100_300, df_100_100_500, df_200_100_300])
df_tot['x'] = 0
for n in N: df_tot['x'] += df_tot[f'x_{n+1}']
df_tot['y'] = 0
for p in P: df_tot['y'] += df_tot[f'y_{p}']
df_tot = df_tot.sort_values(by=['x','y'])

fig = px.scatter(df_tot, x='x', y='avg_cost', color='type')
fig.update_traces(marker={'size': 5})
fig.show(renderer='browser')

fig = px.scatter(df_tot, x='y', y='avg_cost', color='type')
fig.update_traces(marker={'size': 5})
fig.show(renderer='browser')

fig = px.scatter_3d(df_tot, x='x', y='y', z='avg_cost', color='type')
fig.update_traces(marker={'size': 4})
fig.show(renderer='browser')

# %% Fit Neural Network Model and View Predictions
layers_bounds = (2, 6)
neurons_bounds = (1, 20) 

# for i in range(100):
#     layers = np.random.randint(layers_bounds[0],layers_bounds[1]+1)
#     neurons = np.random.randint(neurons_bounds[0],neurons_bounds[1]+1)*5
#     nn_layers = [len(N)+len(P)+2] + [neurons for l in range(layers)] + [1]

nn_layers = (len(N)+len(P)+2,     40,40,40,40,40,   1)

# Initialize Data
x_cols = df_tot.drop(['disc_cost','avg_cost', 'type'], axis=1)
y_cols = df_tot['avg_cost']

# Split into train and test
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_cols, y_cols, test_size = 0.2, train_size = 0.8)

# Create the model
model = Sequential()
for layer in range(len(nn_layers) - 1):
    model.add(Dense(nn_layers[layer+1], input_dim=nn_layers[layer], activation="relu", name=f'layer_{layer}', kernel_regularizer=regularizers.l1_l2(l1=0.1, l2=0.1)))
    # if layer != len(nn_layers)-2: model.add(Dropout(rate = 0.01))

# Fit the model
model.compile(loss='MeanSquaredError', optimizer='adam', metrics=['MeanAbsolutePercentageError'])
history = model.fit(x=x_train, y=y_train, epochs=500, validation_data=(x_test, y_test), verbose=0, callbacks=[TqdmCallback(verbose=1)])  
# history = model.fit(x=x_train, y=y_train, epochs=150, validation_data=(x_test, y_test), verbose=0)    

print(f"nn structure: {nn_layers}, final loss: {history.history['val_mean_absolute_percentage_error'][-1]}")

# %% Plotting
# Predictions
y_pred = model.predict(x_cols).flatten()

# Prediction on a overall graph
fig = px.scatter_3d(df_tot, x='x', y='y', z='avg_cost', color='type')
fig.add_scatter3d(x=df_tot['x'], y=df_tot['y'], z=y_pred, mode='markers')
fig.update_traces(marker=dict(size=4))
fig.show(renderer='browser')

# Prediciton vs fitted
fig = px.scatter(x=df_tot['avg_cost'], y=y_pred, opacity=0.7)
fig.add_scatter(x=[0, df_tot['avg_cost'].max()], y=[0, max(y_pred)], mode='lines')
fig.update_traces(marker=dict(size=5))
fig.show(renderer='browser')

# Predictions generalization on overall graph
rs = np.random.RandomState(seed = 0)
nums = 3000

st_x = {f"x_{n+1}":rs.randint(0, int(data['capacity'])+1, nums) for n in N}
p_bound = {p: df_tot[f'y_{p}'].max()+1 for p in P}
st_y = {f"y_{p}":rs.randint(0, p_bound[p], nums) for p in P}

df_test = pd.DataFrame.from_dict({**st_y, **st_x})
df_test['x'] = 0
for n in N: df_test['x'] += df_test[f'x_{n+1}']
df_test['y'] = 0
for p in P: df_test['y'] += df_test[f'y_{p}']
df_test = pd.concat([df_test, df_tot.drop(columns = ['type','disc_cost','avg_cost'])])
df_test = df_test.sort_values(by=['x','y'])


y_pred = model.predict(df_test).flatten()
fig = px.scatter_3d(df_tot, x='x', y='y', z='avg_cost', color='type')
fig.add_scatter3d(x=df_test['x'], y=df_test['y'], z=y_pred, mode='markers')
fig.update_traces(marker=dict(size=3))
fig.show(renderer='browser')


# %%
