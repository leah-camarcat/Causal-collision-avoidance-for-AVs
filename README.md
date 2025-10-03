# Causal Collision Avoidance for Autonomous Vehicles

## Overview
This repository contains code and experiments for a **structural causal model (SCM)** used to avoid collisions in autonomous driving. The model uses counterfactual reasoning to decide which emergency evasive manoeuvre to take. Several versions of SCMs are implemented with the idea of safety distance to estimate the imminent collision outcome: 
- 1D deterministic SCM (longitudinal collision avoidance) 
- 2D deterministic SCM (longitudinal and lateral collision avoidance)
- 2D learned SCM (longitudinal and lateral collision avoidance)

---

## Features
- Makes use of Waymo example tfrecord data and the corresponding Waymax simulator
- Simulation of selected scenarios
- Safety monitoring module providing actionable recommendations:
  - Acceleration / deceleration
  - Swerving / lane adjustment
- Two test cases are implemented: 
  - Sudden braking of the leading vehicle
  - Sudden close cut in of a leading adjacent vehicle
---

## Data download
- We use Waymo data, please visit their website to download the tfrecords
- Please use this file structure to store the tfrecord files: 
  |- data
    |- motion_v_1_3_0
      |- uncompressed
        |- tf_example
          |- training

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Causal-collision-avoidance-for-AVs.git

# Navigate into the repo
cd Causal-collision-avoidance-for-AVs

# Install required Python packages
pip install -r requirements.txt


