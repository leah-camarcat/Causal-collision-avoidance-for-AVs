"""
Training script for Causal CNN Risk Model on Waymax data.
Uses MTTC-based ground truth and multi-agent observations.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..",  "..", "..")))
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import dataclasses
from itertools import islice
import re
import tensorflow as tf

from waymax.agents.causal_cnn.causal_cnn_model import CausalRiskCNN
from waymax.agents.causal_cnn.ground_truth_mttc import (
    create_mttc_risk_grid,
    extract_multi_agent_observations
)
from waymax import config as _config, dataloader
import wandb

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_step(state, batch, model, rng):
    """Single training step."""
    def loss_fn(params):
        dropout_rng = jax.random.fold_in(rng, state.step)
        predictions, _ = model.apply(
            params, 
            batch['observations'], 
            training=True,
            rngs={'dropout': dropout_rng}
            )
        
        # Binary cross-entropy loss
        bce_loss = optax.sigmoid_binary_cross_entropy(predictions, batch['risk_labels'])
        
        # Spatial smoothness regularization
        dx = predictions[:, 1:, :, :] - predictions[:, :-1, :, :]
        dy = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
        smoothness_loss = jnp.mean(dx**2) + jnp.mean(dy**2)
        
        total_loss = jnp.mean(bce_loss) + 0.01 * smoothness_loss
        
        return total_loss, {
            'bce': jnp.mean(bce_loss),
            'smoothness': smoothness_loss,
            'total': total_loss
        }
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)

    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_causal_risk_cnn(
    tfrecord_pattern: str,
    filtered_scenarios_dir: str,
    output_path: str = "./trained_risk_model.pkl",
    grid_size: int = 64,
    grid_range: float = 50.0,
    history_length: int = 10,
    max_agents: int = 8,
    max_num_objects: int = 32,
    epochs_per_scenario: int = 3,
    learning_rate: float = 1e-4,
    save_every: int = 100,
    use_mttc: bool = True
):
    """
    Train the causal CNN risk model on filtered Waymax scenarios.
    
    Args:
        tfrecord_pattern: Glob pattern for TFRecord files
            e.g., "../../data/motion_v_1_3_0/.../training_tfexample.tfrecord-*"
        filtered_scenarios_dir: Directory containing filtered scenario .pkl files
            e.g., "../cutin_filtered_data"
        output_path: Where to save the trained model
        grid_size: Spatial resolution of risk grid
        grid_range: Spatial range in meters (±grid_range)
        history_length: Number of past timesteps to use
        max_agents: Number of surrounding agents to consider
        max_num_objects: Max objects in Waymax config
        epochs_per_scenario: Training epochs per scenario
        learning_rate: Adam learning rate
        save_every: Save checkpoint every N scenarios
        use_mttc: Use MTTC ground truth (vs simple Gaussian)
    """
    
    print("="*70)
    print("TRAINING CAUSAL CNN RISK MODEL (FILTERED SCENARIOS)")
    print("="*70)
    print(f"TFRecord Pattern: {tfrecord_pattern}")
    print(f"Filtered Scenarios Dir: {filtered_scenarios_dir}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"Grid Range: ±{grid_range}m")
    print(f"History Length: {history_length} timesteps")
    print(f"Max Agents: {max_agents} surrounding vehicles")
    print(f"Ground Truth: {'MTTC-based' if use_mttc else 'Gaussian'}")
    print("="*70)
    
    # Load filtered scenario IDs
    filtered_scenarios = [
        f[:-4] for f in os.listdir(filtered_scenarios_dir) 
        if f.endswith('.pkl')
    ]
    print(f"Found {len(filtered_scenarios)} filtered scenarios")
    
    # Track processed scenarios
    processed_scenarios = set()
    pending_scenarios = set(filtered_scenarios)
    
    # Get TFRecord files
    tfrecord_files = tf.io.gfile.glob(tfrecord_pattern)
    tfrecord_files = sorted(
        tfrecord_files,
        key=lambda x: int(re.search(r'tfrecord-(\d+)-of', x).group(1))
    )
    print(f"Found {len(tfrecord_files)} TFRecord shards")
    
    # Initialize model with multi-agent input
    obs_features = max_agents * 6
    
    model = CausalRiskCNN(
        grid_size=grid_size,
        grid_range=grid_range,
        history_length=history_length
    )
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    dummy_obs = jnp.ones((1, history_length, obs_features))
    params = model.init(rng, dummy_obs, training=False)
    
    # Create optimizer and training state
    optimizer = optax.adam(learning_rate)
    train_state_obj = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # JIT compile training step
    #jit_train_step = jax.jit(lambda s, b: train_step(s, b, model))
    train_rng = jax.random.PRNGKey(123)
    
    # Training metrics
    all_losses = []
    scenario_count = 0
    
    # ============================================================================
    # MAIN TRAINING LOOP OVER SHARDS
    # ============================================================================
    
    print("\nStarting training loop...")
    
    for shard_idx, shard_file in enumerate(tfrecord_files):
        if len(pending_scenarios) == 0:
            print("All filtered scenarios processed!")
            break
        
        print(f"\n{'='*70}")
        print(f"Processing Shard {shard_idx + 1}/{len(tfrecord_files)}")
        print(f"File: {shard_file.split('/')[-1]}")
        print(f"Remaining scenarios: {len(pending_scenarios)}")
        print(f"{'='*70}")
        
        # Configure for this shard
        config = dataclasses.replace(
            _config.WOD_1_3_0_TRAIN,
            path=shard_file,
            max_num_objects=max_num_objects
        )
        
        # Count scenarios in shard
        all_counts = sum(1 for _ in tf.data.TFRecordDataset(shard_file))
        print(f"Shard contains {all_counts} scenarios")
        
        # Create data iterator for this shard
        data_iter = dataloader.simulator_state_generator(config=config)
        
        # Process each scenario in shard
        with tqdm(total=all_counts, desc=f"Shard {shard_idx+1}") as pbar:
            for scenario_idx in range(all_counts):
                try:
                    # Get scenario
                    scenario = next(islice(data_iter, scenario_idx, scenario_idx + 1))
                    scenario_id = scenario.object_metadata.scenario_id[0].decode('utf-8')
                    
                    # Check if this scenario should be processed
                    if scenario_id not in filtered_scenarios:
                        pbar.update(1)
                        continue
                    
                    if scenario_id in processed_scenarios:
                        pbar.update(1)
                        continue
                    
                    # Mark as processed
                    processed_scenarios.add(scenario_id)
                    pending_scenarios.discard(scenario_id)
                    
                    # Find ego vehicle
                    is_sdc_mask = scenario.object_metadata.is_sdc
                    if not jnp.any(is_sdc_mask):
                        pbar.update(1)
                        continue
                    
                    ego_idx = jnp.where(is_sdc_mask)[0][0]
                    
                    # Train on multiple timesteps from this scenario
                    num_timesteps = scenario.log_trajectory.valid.shape[1]
                    num_samples_this_scenario = 0
                    
                    for epoch in range(epochs_per_scenario):
                        for t in range(history_length, min(num_timesteps - 1, history_length + 20)):
                            # Check ego validity
                            if not scenario.log_trajectory.valid[ego_idx, t]:
                                continue
                            
                            # Create state at timestep t
                            temp_state = scenario.replace(timestep=t)
                            
                            # Extract MULTI-AGENT observations
                            observations = extract_multi_agent_observations(
                                temp_state, ego_idx, history_length, max_agents
                            )
                            
                            # Create ground truth risk grid using MTTC
                            risk_labels = create_mttc_risk_grid(
                                temp_state, ego_idx, grid_size, grid_range
                            )
                        
                            # Prepare batch
                            batch = {
                                'observations': observations[None, ...],
                                'risk_labels': risk_labels
                            }
                            
                            # Training step
                            #train_state_obj, metrics = jit_train_step(train_state_obj, batch)
                            train_rng, step_rng = jax.random.split(train_rng)
                            train_state_obj, metrics = train_step(train_state_obj, batch, model, step_rng)
                            all_losses.append(float(metrics['total']))
                            num_samples_this_scenario += 1
                        
                        # wandb logging
                        run.log({"epoch":epoch, "bce_loss":metrics['bce'], "total_loss":metrics['total']})
                    scenario_count += 1
                    
                    # Update progress bar with scenario info
                    if num_samples_this_scenario > 0:
                        recent_loss = np.mean(all_losses[-min(50, len(all_losses)):])
                        pbar.set_postfix({
                            'scenarios': scenario_count,
                            'samples': num_samples_this_scenario,
                            'loss': f'{recent_loss:.4f}'
                        })
                    
                    # Save checkpoint
                    if scenario_count % save_every == 0 and scenario_count > 0:
                        checkpoint_path = output_path.replace('.pkl', f'_checkpoint_{scenario_count}.pkl')
                        save_model(
                            train_state_obj.params, checkpoint_path, 
                            grid_size, grid_range, history_length, max_agents, all_losses
                        )
                        tqdm.write(f"✓ Checkpoint saved: {checkpoint_path}")
                        tqdm.write(f"  Processed: {scenario_count}/{len(filtered_scenarios)} scenarios")
                        tqdm.write(f"  Remaining: {len(pending_scenarios)} scenarios")
                
                except Exception as e:
                    tqdm.write(f"✗ Error in scenario {scenario_idx}: {e}")
                    continue
                finally:
                    pbar.update(1)
        
        print(f"Shard {shard_idx + 1} complete. Processed {scenario_count} scenarios so far.")
    
    # ============================================================================
    # TRAINING COMPLETE
    # ============================================================================
    
    # Save final model
    save_model(
        train_state_obj.params, output_path, 
        grid_size, grid_range, history_length, max_agents, all_losses
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total scenarios processed: {scenario_count}/{len(filtered_scenarios)}")
    print(f"Total training samples: {len(all_losses)}")
    print(f"Final loss: {np.mean(all_losses[-100:]):.4f}")
    print(f"Model saved to: {output_path}")
    
    # Save list of processed scenarios for reference
    processed_list_path = output_path.replace('.pkl', '_processed_scenarios.txt')
    with open(processed_list_path, 'w') as f:
        for scenario_id in sorted(processed_scenarios):
            f.write(f"{scenario_id}\n")
    print(f"Processed scenarios list: {processed_list_path}")
    
    # Plot training curve
    plot_training_curve(all_losses, output_path.replace('.pkl', '_training.png'))
    
    # Check if any scenarios were missed
    if pending_scenarios:
        print(f"\nWarning: {len(pending_scenarios)} scenarios were not found in TFRecords:")
        for scenario_id in list(pending_scenarios)[:10]:
            print(f"  - {scenario_id}")
        if len(pending_scenarios) > 10:
            print(f"  ... and {len(pending_scenarios) - 10} more")
    
    return train_state_obj.params


def save_model(params, path, grid_size, grid_range, history_length, max_agents, losses=None):
    """
    Save model parameters and metadata to a checkpoint file.
    
    Args:
        params: Flax parameters (frozen dict)
        path: file path to save (e.g. "./trained_risk_model.pkl")
        grid_size: risk grid size
        grid_range: spatial range
        history_length: number of timesteps used
        max_agents: number of agents used
        losses: optional list of training losses
    """
    state = {
        "params": params,
        "config": {
            "grid_size": grid_size,
            "grid_range": grid_range,
            "history_length": history_length,
            "max_agents": max_agents
        },
        "losses": losses
    }
    with open(path, "wb") as f:
        pickle.dump(state, f)
    print(f"✓ Model checkpoint saved to {path}")


def plot_training_curve(losses, save_path):
    """Plot and save training curve."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, alpha=0.3, label='Loss')
    window = min(100, len(losses) // 10)
    if window > 0:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg, 'r-', linewidth=2, label=f'MA-{window}')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(losses, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(losses), color='r', linestyle='--', label=f'Mean: {np.mean(losses):.4f}')
    plt.xlabel('Loss')
    plt.ylabel('Frequency')
    plt.title('Loss Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    tfrecord_pattern = "data/motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord-*"
    filtered_scenarios_dir = "docs/cutin_filtered_data"
    run = wandb.init(
        entity="leah-camarcat",
        project="Waymax RL",
        config={
            "epochs":100, 
            "version": "v1"
        }
    )
    # Train model on filtered scenarios
    trained_params = train_causal_risk_cnn(
        tfrecord_pattern=tfrecord_pattern,
        filtered_scenarios_dir=filtered_scenarios_dir,
        output_path="waymax/agents/causal_cnn/trained_risk_model.pkl",
        grid_size=64,
        grid_range=50.0,
        history_length=10,
        max_agents=8,
        max_num_objects=32,
        epochs_per_scenario=30,
        learning_rate=1e-4,
        save_every=10,  # Save every 50 scenarios
        use_mttc=True
    )
    run.finish()
    print("\n✓ Training complete!")
    print("Model ready for inference with assess_baseline_risk()")