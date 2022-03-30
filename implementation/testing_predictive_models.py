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
# df_100_50_100 = pd.read_csv('data/simulation-value_100_50_100.csv')
# df_100_50_150 = pd.read_csv('data/simulation-value_100_50_150.csv')
# df_100_50_250 = pd.read_csv('data/simulation-value_100_50_250.csv')
# df_100_50_450 = pd.read_csv('data/simulation-value_100_50_450.csv')
# df_100_50_850 = pd.read_csv('data/simulation-value_100_50_850.csv')

# # Modify Data
# df_100_50_100['type'] = '100_50_100'
# df_100_50_150['type'] = '100_50_150'
# df_100_50_250['type'] = '100_50_250'
# df_100_50_450['type'] = '100_50_450'
# df_100_50_850['type'] = '100_50_850'

# Iterations
df_iter = pd.DataFrame()

# durs = [100, 150, 250, 450, 850]
for i in range(24):
    df_new = pd.read_csv(f'data/sim-optim-logs/simulation-value-iter{i}_300_50_450.csv')
    df_new['type'] = f'iter{i}_300_50_450'
    df_iter = pd.concat([df_iter, df_new])

# %% Graph Data
# Review Data
# df_tot = pd.concat([df_100_50_100, df_100_50_150, df_100_50_250, df_100_50_450, df_100_50_850])
df_tot = df_iter
df_tot['x'] = 0
for n in N: df_tot['x'] += df_tot[f'x_{n+1}']
df_tot['y'] = 0
for p in P: df_tot['y'] += df_tot[f'y_{p}']
df_tot = df_tot.sort_values(by=['x','y'])

fig = px.scatter(df_tot, x='x', y='disc_cost', color='type')
fig.update_traces(marker={'size': 5})
fig.show(renderer='browser')

fig = px.scatter(df_tot, x='y', y='disc_cost', color='type')
fig.update_traces(marker={'size': 5})
fig.show(renderer='browser')

fig = px.scatter_3d(df_tot, x='x', y='y', z='disc_cost', color='type')
fig.update_traces(marker={'size': 4})
fig.show(renderer='browser')

fig = px.scatter_3d(df_tot, x='x', y='y', z='avg_cost', color='type')
fig.update_traces(marker={'size': 4})
fig.show(renderer='browser')


# %% Fit Regression Model 

x_cols = df_tot.drop(['disc_cost','avg_cost', 'type'], axis=1)
y_cols = df_tot['avg_cost']
reg = linear_model.Ridge(alpha=0.5)
reg.fit(x_cols, y_cols)

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
# y_pred = model.predict(x_cols).flatten()
y_pred = reg.predict(x_cols).flatten()

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
# rs = np.random.RandomState(seed = 0)
# nums_tot = 10000
# p_max = df_tot['y'].max()
# n_max = df_tot['x'].max()

# # X values
# st_x = {f"x_{n+1}":[] for n in N}
# x_val = np.random.randint(0, n_max+1, size=nums_tot)
# for x_val_i in x_val:
#     x_vals = np.random.multinomial(x_val_i, np.ones(len(N))/len(N))
#     for n in N: st_x[f'x_{n+1}'].append(x_vals[n])

# # Y Values
# st_y = {f"y_{p}":[] for p in P}
# y_val = np.random.randint(0, p_max+1, size=nums_tot)
# for y_val_i in y_val:
#     y_vals = np.random.multinomial(y_val_i, np.ones(len(P))/len(P))
#     for p in P: st_y[f'y_{p}'].append(y_vals[P.index(p)])

# # Create Dataframe
# df_test = pd.DataFrame.from_dict({**st_y, **st_x})
# df_test['x'] = 0
# for n in N: df_test['x'] += df_test[f'x_{n+1}']
# df_test['y'] = 0
# for p in P: df_test['y'] += df_test[f'y_{p}']
# df_test = pd.concat([df_test, df_tot.drop(columns = ['type','disc_cost','avg_cost'])])
# df_test = df_test.sort_values(by=['x','y'])

# # Prediction
# # # y_pred = model.predict(df_test).flatten()
# y_pred = reg.predict(df_test).flatten()
# fig = px.scatter_3d(df_tot, x='x', y='y', z='avg_cost', color='type')
# fig.add_scatter3d(x=df_test['x'], y=df_test['y'], z=y_pred, mode='markers')
# fig.update_traces(marker=dict(size=3))
# fig.show(renderer='browser')


# %%
