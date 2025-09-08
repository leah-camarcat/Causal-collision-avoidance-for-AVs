
import numpy as np
import imageio
from tqdm import tqdm
import dataclasses

from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import visualization
from itertools import islice
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

config = dataclasses.replace(_config.WOD_1_3_0_TRAIN_EX, max_num_objects=32)
data_iter = dataloader.simulator_state_generator(config=config)
scenario = next(islice(data_iter, 4, 5))

# Using logged trajectory
img = visualization.plot_simulator_state(scenario, use_log_traj=True)

imgs = []

state = scenario
for _ in range(scenario.remaining_timesteps):
  state = datatypes.update_state_by_log(state, num_steps=1)
  imgs.append(visualization.plot_simulator_state(state, use_log_traj=True))


with imageio.get_writer('output.mp4', fps=10) as writer:
  for frame in imgs:
    writer.append_data(frame)