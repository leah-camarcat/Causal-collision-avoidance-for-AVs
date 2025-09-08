import numpy as np
import pickle
import pandas as pd

def compute_acc(scenario_data):
    ego_data = scenario_data[0]
    ego_ax = (np.diff(ego_data[2, :], axis=0)/0.1).reshape((90,1))
    ego_ay = (np.diff(ego_data[3, :], axis=0)/0.1).reshape((90,1))
    ego_acc = np.concatenate((ego_ax, ego_ay), axis=1)
    ego_acc = np.concatenate((ego_acc[0, :].reshape(1,2), ego_acc))
    ego_data = np.concatenate((np.transpose(ego_data), ego_acc), axis=1)

    agent_data = scenario_data[1]
    agent_ax = (np.diff(agent_data[2, :], axis=0)/0.1).reshape((90,1))
    agent_ay = (np.diff(agent_data[3, :], axis=0)/0.1).reshape((90,1))
    agent_acc = np.concatenate((agent_ax, ego_ay), axis=1)
    agent_acc = np.concatenate((agent_acc[0, :].reshape(1,2), agent_acc))
    agent_data = np.concatenate((np.transpose(agent_data), agent_acc), axis=1)
    return ego_data, agent_data


def y_func(speedL, accL, speedF, headway, accF, r=0.7):
    if (accL < 0) and (accL > - 0.1):
        y = speedF * headway - r * speedF - (speedF**2)/(2*accF)
    else: 
        y = speedF*headway + (speedL**2)/(2*accL) \
         - r * speedF - (speedF**2) / (2*accF)
    return y


with open('September_2025/1fb44c31801c956d_m.pkl', 'rb') as f:
    scenario_data = pickle.load(f)

valid = 1
ego_data, agent_data = compute_acc(scenario_data)
for t in range(10,90):
    speedL = np.linalg.norm([agent_data[t, 2:4]])
    accL = np.linalg.norm([agent_data[t, 4:]])
    speedF = np.linalg.norm([ego_data[t, 2:4]])
    headway = np.linalg.norm([agent_data[t, :2]]) - np.linalg.norm([ego_data[t, :2]])
    if valid == 1:
        accF = np.linalg.norm([ego_data[t, 4:]])
    y = y_func(speedL, accL, speedF, headway, accF)
    step = 0.001
    if y < 0:
        if accF > 0:
            accF = 0
        print(y)
        iter = 0
        accF_try = accF
        y_new = y
        while (y_new < 0) and (iter < 1000):
            accF_try = accF_try - step
            y_new = y_func(speedL, accL, speedF, headway, accF_try)
            iter += 1
        print(f'Initial deceleration: {accF}, new deceleration: {accF_try} at t={t}')
        valid = 0
        accF = accF_try
    else:
        print(f'All good at time {t}')
        valid = 1
        continue
