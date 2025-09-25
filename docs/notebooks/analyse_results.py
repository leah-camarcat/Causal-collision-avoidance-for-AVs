import pandas as pd
import numpy as np

results_IDM = pd.read_csv("docs/notebooks/testing_IDM_breakdown_dry.csv")
results_davis = pd.read_csv("docs/notebooks/testing_davis_breakdown_dry.csv")
N = len(results_IDM[results_IDM.col_outcome.isna()==False])
success_rate_IDM = results_IDM.col_outcome.sum()/N
success_rate_davis = results_davis.col_outcome.sum()/N
print(f'success rate IDM: {success_rate_IDM}, david: {success_rate_davis}')
failed_scenarios_IDM = results_IDM[pd.isna(results_IDM.col_outcome)]['Unnamed: 0']
failed_scenarios_davis = results_davis[pd.isna(results_davis.col_outcome)]['Unnamed: 0']

print(failed_scenarios_davis.isin(failed_scenarios_IDM))
