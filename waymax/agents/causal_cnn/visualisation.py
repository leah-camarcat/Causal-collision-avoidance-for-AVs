"""
Visualization utilities for overlaying CNN risk grids on Waymax scenarios.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import jax.numpy as jnp
from waymax import datatypes, visualization
from waymax.agents.causal_cnn.ground_truth_mttc import create_mttc_risk_grid, compute_mttc_vectorized
from waymax.visualization import viz
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap


def visualize_risk_grid_overlay(
    state: datatypes.SimulatorState,
    ego_idx: int,
    grid_range_long: float = 50.0,
    grid_range_lat: float = 15.0,
    ax=None,
    show_vehicles: bool = True,
    show_trajectories: bool = True,
    show_mttc_values: bool = True
):
    """
    Visualize risk grid overlaid on Waymax scenario.
    
    Args:
        state: Current simulator state
        ego_idx: Ego vehicle index
        grid_range_long: Longitudinal range (±meters)
        grid_range_lat: Lateral range (±meters)
        ax: Matplotlib axis (creates new if None)
        show_vehicles: Draw vehicle bounding boxes
        show_trajectories: Draw predicted trajectories
        show_mttc_values: Annotate MTTC values above vehicles
    
    Returns:
        fig, ax: Matplotlib figure and axis
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = ax.figure
    
    # === STEP 1: Generate risk grid ===
    risk_grid = create_mttc_risk_grid(
        state, ego_idx, 
        grid_size=64,
        grid_range_long=grid_range_long,
        grid_range_lat=grid_range_lat
    )
    risk_grid_2d = risk_grid[0, :, :, 0]  # Remove batch/channel dims
    
    # === STEP 2: Get ego vehicle info ===
    current_timestep = state.timestep
    ego_pos = state.sim_trajectory.xy[ego_idx, current_timestep]
    ego_yaw = state.sim_trajectory.yaw[ego_idx, current_timestep]
    ego_length = state.sim_trajectory.length[ego_idx, current_timestep]
    ego_width = state.sim_trajectory.width[ego_idx, current_timestep]
    
    # Ego coordinate frame
    ego_forward = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
    ego_lateral = np.array([-np.sin(ego_yaw), np.cos(ego_yaw)])
    
    # === STEP 3: Create grid coordinates in world frame ===
    # Grid is in ego-centric coordinates, need to transform to world
    grid_size = risk_grid_2d.shape[0]
    
    # Grid cell centers in ego frame
    long_coords = np.linspace(-grid_range_long, grid_range_long, grid_size)
    lat_coords = np.linspace(-grid_range_lat, grid_range_lat, grid_size)
    
    # Transform grid corners to world coordinates for plotting
    grid_corners_ego = np.array([
        [-grid_range_long, -grid_range_lat],
        [grid_range_long, -grid_range_lat],
        [grid_range_long, grid_range_lat],
        [-grid_range_long, grid_range_lat]
    ])
    
    # Rotate and translate to world frame
    grid_corners_world = ego_pos + grid_corners_ego @ np.stack([ego_forward, ego_lateral]).T
    
    # === STEP 4: Plot risk grid as heatmap ===
    # Create custom colormap (transparent low risk → red high risk)
    colors = ['#00FF0000', '#FFFF0080', '#FF8000C0', '#FF0000FF']  # RGBA
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('risk', colors, N=n_bins)
    
    # Calculate extent in world coordinates (oriented with ego)
    # Need to create transformed image
    extent = [
        ego_pos[0] - grid_range_long * np.cos(ego_yaw) - grid_range_lat * np.sin(ego_yaw),
        ego_pos[0] + grid_range_long * np.cos(ego_yaw) + grid_range_lat * np.sin(ego_yaw),
        ego_pos[1] - grid_range_long * np.sin(ego_yaw) + grid_range_lat * np.cos(ego_yaw),
        ego_pos[1] + grid_range_long * np.sin(ego_yaw) - grid_range_lat * np.cos(ego_yaw)
    ]
    
    # For simplicity, if yaw is small, plot directly
    # For general case, would need to rotate the image
    if np.abs(ego_yaw) < 0.1:  # Nearly aligned with world axes
        extent_simple = [
            ego_pos[0] - grid_range_long, ego_pos[0] + grid_range_long,
            ego_pos[1] - grid_range_lat, ego_pos[1] + grid_range_lat
        ]
        ax.imshow(
            risk_grid_2d.T, 
            extent=extent_simple,
            origin='lower',
            cmap=cmap,
            alpha=0.6,
            vmin=0, vmax=1,
            zorder=1
        )
    else:
        # For rotated case, plot grid outline
        grid_poly = patches.Polygon(
            grid_corners_world,
            fill=False,
            edgecolor='yellow',
            linewidth=2,
            linestyle='--',
            zorder=1,
            label='Risk Grid Bounds'
        )
        ax.add_patch(grid_poly)
        
        # Plot risk as colored rectangles (more expensive but works for any rotation)
        cell_size_long = 2 * grid_range_long / grid_size
        cell_size_lat = 2 * grid_range_lat / grid_size
        
        # Sample every Nth cell for performance
        #step = max(1, grid_size // 32)
        step = 1
        for i in range(0, grid_size, step):
            for j in range(0, grid_size, step):
                risk_val = risk_grid_2d[i, j]
                if risk_val < 0.05:  # Skip very low risk
                    continue
                
                # Cell center in ego frame
                long_ego = long_coords[i]
                lat_ego = lat_coords[j]
                
                # Transform to world
                cell_center_world = ego_pos + long_ego * ego_forward + lat_ego * ego_lateral
                
                # Draw colored circle
                color = cmap(risk_val)
                circle = patches.Circle(
                    cell_center_world,
                    radius=max(cell_size_long, cell_size_lat) * step,
                    color=color,
                    alpha=0.5,
                    zorder=1
                )
                ax.add_patch(circle)
    
    # === STEP 5: Draw vehicles ===
    if show_vehicles:
        # Compute MTTC values for annotation
        if show_mttc_values:
            mttc_risks = compute_mttc_vectorized(state, ego_idx)
        
        for obj_idx in range(state.sim_trajectory.num_objects):
            if not state.object_metadata.is_valid[obj_idx]:
                continue
            
            pos = state.sim_trajectory.xy[obj_idx, current_timestep]
            yaw = state.sim_trajectory.yaw[obj_idx, current_timestep]
            length = state.sim_trajectory.length[obj_idx, current_timestep]
            width = state.sim_trajectory.width[obj_idx, current_timestep]
            
            # Draw vehicle rectangle
            if obj_idx == ego_idx:
                color = 'blue'
                linewidth = 3
                label = 'Ego'
            else:
                color = 'black'
                linewidth = 2
                label = None
            
            # Vehicle corners (centered at pos)
            corners_local = np.array([
                [-length/2, -width/2],
                [length/2, -width/2],
                [length/2, width/2],
                [-length/2, width/2]
            ])
            
            # Rotate and translate
            rot_mat = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
            corners_world = pos + corners_local @ rot_mat.T
            
            vehicle_rect = patches.Polygon(
                corners_world,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                zorder=3,
                label=label
            )
            ax.add_patch(vehicle_rect)
            
            # Draw heading arrow
            arrow_length = length * 0.6
            arrow_end = pos + arrow_length * np.array([np.cos(yaw), np.sin(yaw)])
            ax.arrow(
                pos[0], pos[1],
                arrow_end[0] - pos[0], arrow_end[1] - pos[1],
                head_width=width*0.4,
                head_length=length*0.2,
                fc=color,
                ec=color,
                zorder=4,
                alpha=0.7
            )
            
            # Annotate MTTC value
            if show_mttc_values and obj_idx != ego_idx:
                risk_val = mttc_risks[obj_idx]
                if risk_val > 0.01:
                    ax.text(
                        pos[0], pos[1] + width,
                        f'R={risk_val:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        color='red' if risk_val > 0.5 else 'orange',
                        weight='bold',
                        zorder=5,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
                    )
    
    # === STEP 6: Draw predicted trajectories ===
    if show_trajectories:
        for obj_idx in range(state.sim_trajectory.num_objects):
            if not state.object_metadata.is_valid[obj_idx]:
                continue
            
            pos = state.sim_trajectory.xy[obj_idx, current_timestep]
            vel = np.array([
                state.sim_trajectory.vel_x[obj_idx, current_timestep],
                state.sim_trajectory.vel_y[obj_idx, current_timestep]
            ])
            
            # Simple constant velocity prediction
            future_times = np.linspace(0, 5, 20)  # 5 seconds
            future_pos = pos + vel * future_times[:, None]
            
            if obj_idx == ego_idx:
                ax.plot(future_pos[:, 0], future_pos[:, 1], 
                       'b--', linewidth=1.5, alpha=0.5, zorder=2)
            else:
                ax.plot(future_pos[:, 0], future_pos[:, 1], 
                       'k--', linewidth=1, alpha=0.3, zorder=2)
    
    # === STEP 7: Formatting ===
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title(f'Risk Grid Overlay - Timestep {current_timestep}', fontsize=14, weight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # Set view to focus on ego vehicle area
    margin = 60
    ax.set_xlim(ego_pos[0] - margin, ego_pos[0] + margin)
    ax.set_ylim(ego_pos[1] - margin, ego_pos[1] + margin)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Collision Risk', fontsize=11)
    plt.savefig('docs/risk_map_example.png')
    plt.show()
    return fig, ax


def animate_scenario_with_risk(
    states_list,
    ego_idx: int,
    save_path: str = None,
    grid_range_long: float = 50.0,
    grid_range_lat: float = 15.0
):
    """
    Create animation of scenario with evolving risk grid.
    
    Args:
        states_list: List of SimulatorState objects (trajectory)
        ego_idx: Ego vehicle index
        save_path: Path to save animation (e.g., 'risk_animation.mp4')
        grid_range_long: Longitudinal grid range
        grid_range_lat: Lateral grid range
    
    Returns:
        Animation object
    """
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    def update_frame(frame_idx):
        ax.clear()
        state = states_list[frame_idx]
        visualize_risk_grid_overlay(
            state, ego_idx,
            grid_range_long=grid_range_long,
            grid_range_lat=grid_range_lat,
            ax=ax,
            show_vehicles=True,
            show_trajectories=True,
            show_mttc_values=True
        )
        return ax,
    
    anim = FuncAnimation(
        fig, update_frame,
        frames=len(states_list),
        interval=100,  # 100ms = 10 FPS
        blit=False
    )
    
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=10, dpi=100)
        print(f"Animation saved to {save_path}")
    
    return anim


def plot_simulator_state_with_risk(
    state: datatypes.SimulatorState,
    ego_idx: int,
    use_log_traj: bool = False,
    grid_range_long: float = 50.0,
    grid_range_lat: float = 15.0,
    show_mttc_values: bool = False,
    figsize: tuple = (10, 10)
) -> np.ndarray:
    """
    Enhanced version of visualization.plot_simulator_state with risk grid overlay.
    
    Returns image as numpy array, compatible with imageio for video creation.
    
    Args:
        state: Current simulator state
        ego_idx: Ego vehicle index
        use_log_traj: Use logged trajectory (passed to Waymax viz)
        grid_range_long: Longitudinal grid range (±meters)
        grid_range_lat: Lateral grid range (±meters)
        show_mttc_values: Show risk values above vehicles
        figsize: Figure size
    
    Returns:
        img: RGB image as numpy array (H, W, 3)
    """
    
    # === STEP 1: Get Waymax's base visualization ===
    base_img = visualization.plot_simulator_state(state, use_log_traj=use_log_traj)
    
    # === STEP 2: Create figure and display base image ===
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(base_img)
    ax.axis('off')
    
    # Get current axis limits from base visualization
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # === STEP 3: Generate and overlay risk grid ===
    risk_grid = create_mttc_risk_grid(
        state, ego_idx,
        grid_size=64,
        grid_range_long=grid_range_long,
        grid_range_lat=grid_range_lat
    )
    risk_grid_2d = risk_grid[0, :, :, 0]
    
    # Get ego info
    current_timestep = state.timestep
    ego_pos_x = state.sim_trajectory.x[ego_idx, current_timestep]
    ego_po_y = state.sim_trajectory.y[ego_idx, current_timestep]
    ego_pos = [ego_pos_x, ego_po_y]
    ego_yaw = state.sim_trajectory.yaw[ego_idx, current_timestep]
    print(f"Risk grid range: [{risk_grid_2d.min():.3f}, {risk_grid_2d.max():.3f}]")
    print(f"Non-zero cells: {np.count_nonzero(risk_grid_2d)}")
    
    # Create transparent colormap
    colors = ['#00000000', '#FFFF0060', '#FF8000A0', '#FF0000D0']
    cmap = LinearSegmentedColormap.from_list('risk', colors, N=100)
    
    
    if len(all_positions) > 0:
        all_positions = np.array(all_positions)
        world_x_min = all_positions[:, 0].min()
        world_x_max = all_positions[:, 0].max()
        world_y_min = all_positions[:, 1].min()
        world_y_max = all_positions[:, 1].max()
        
        # Add padding (Waymax typically adds padding around objects)
        padding = 20  # meters
        world_x_min -= padding
        world_x_max += padding
        world_y_min -= padding
        world_y_max += padding
    else:
        # Fallback: assume 100m x 100m around ego
        world_x_min = ego_pos[0] - 50
        world_x_max = ego_pos[0] + 50
        world_y_min = ego_pos[1] - 50
        world_y_max = ego_pos[1] + 50
    
    world_width = world_x_max - world_x_min
    world_height = world_y_max - world_y_min
    
    print(f"World bounds: X=[{world_x_min:.1f}, {world_x_max:.1f}], Y=[{world_y_min:.1f}, {world_y_max:.1f}]")
    print(f"World size: {world_width:.1f}m x {world_height:.1f}m")
    print(f"Image size: {img_width}px x {img_height}px")
    
    # === STEP 4: Calculate conversion factors ===
    # Pixels per meter in each dimension
    px_per_meter_x = img_width / world_width
    px_per_meter_y = img_height / world_height

    def world_to_pixel(world_x, world_y, padding=20):
        world_x_min = - padding
        world_y_min = + padding
        """Convert world coordinates to pixel coordinates."""
        # X: left to right
        pixel_x = (world_x - world_x_min) * px_per_meter_x
        # Y: top to bottom (NOTE: image Y is inverted!)
        pixel_y = img_height - (world_y - world_y_min) * px_per_meter_y
        return pixel_x, pixel_y
    ego_px_x, ego_px_y = world_to_pixel(ego_pos[0], ego_pos[1])
    print(f"Ego position (pixels): ({ego_px_x:.1f}, {ego_px_y:.1f})")

    margin = 0.1  # 10% margin on each side
    extent_pixels = [
        ego_px_x - 50,           # left
        ego_px_x + 50,     # right  
        ego_px_y - 83,    # bottom (note: image coordinates are flipped)
        ego_px_y + 83     # top
    ]
    
    im = ax.imshow(
        #risk_grid_2d.T[::-1, :],  # or risk_grid_2d depending on orientation
        risk_grid_2d,
        origin='lower',
        cmap='hot',
        alpha=0.6,  # Semi-transparent
        vmin=0.0,
        vmax=1.0,
        extent=extent_pixels,  # Adjust to your grid_range
        zorder=10  # Draw on top
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Risk Level', rotation=270, labelpad=15)
    
    # === STEP 4: Add MTTC value annotations ===
    if show_mttc_values:
        mttc_risks = compute_mttc_vectorized(state, ego_idx)
        
        for obj_idx in range(state.sim_trajectory.num_objects):
            if obj_idx == ego_idx or not state.object_metadata.is_valid[obj_idx]:
                continue
            
            risk_val = mttc_risks[obj_idx]
            if risk_val > 0.05:
                pos = state.sim_trajectory.xy[obj_idx, current_timestep]
                width = state.sim_trajectory.width[obj_idx, current_timestep]
                
                ax.text(
                    pos[0], pos[1] + width + 2,
                    f'{risk_val:.2f}',
                    ha='center', va='bottom',
                    fontsize=10,
                    color='white',
                    weight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.4',
                        facecolor='red' if risk_val > 0.7 else 'orange',
                        alpha=0.8,
                        edgecolor='white',
                        linewidth=1.5
                    )
                )
    
    # Restore original axis limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # === STEP 5: Convert to numpy array ===
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = img[:, :, :3] 
    
    plt.close(fig)
    
    return img


def plot_simulator_state_with_risk_v2(
    state: datatypes.SimulatorState,
    ego_idx: int,
    use_log_traj: bool = False,
    grid_range_long: float = 50.0,
    grid_range_lat: float = 15.0,
    figsize: tuple = (12, 10)
) -> np.ndarray:
    """
    Overlay risk grid on Waymax base image - simplified version.
    """
    # === STEP 1: Get Waymax's base visualization ===
    base_img = visualization.plot_simulator_state(state, use_log_traj=use_log_traj)
    
    # === STEP 2: Generate risk grid ===
    risk_grid = create_mttc_risk_grid(
        state, ego_idx,
        grid_size=64,
        grid_range_long=grid_range_long,
        grid_range_lat=grid_range_lat
    )
    risk_grid_2d = risk_grid[0, :, :, 0]
    
    print(f"Risk grid shape: {risk_grid_2d.shape}")
    print(f"Risk range: [{risk_grid_2d.min():.3f}, {risk_grid_2d.max():.3f}]")
    print(f"Non-zero cells: {np.count_nonzero(risk_grid_2d)}")
    
    # === STEP 3: Create figure with base image ===
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(base_img)
    ax.axis('off')
    
    # === STEP 4: Get ego state ===
    current_timestep = state.timestep
    ego_pos = state.sim_trajectory.xy[ego_idx, current_timestep]
    ego_yaw = state.sim_trajectory.yaw[ego_idx, current_timestep]
    
    print(f"Ego position: {ego_pos}, yaw: {np.degrees(ego_yaw):.1f}°")
    
    # === STEP 5: Manually create rotated risk overlay ===
    # We need to convert world coordinates to pixel coordinates
    # This is approximate - you may need to tune based on Waymax's coordinate system
    
    from matplotlib.colors import LinearSegmentedColormap
    import scipy.ndimage
    
    # Create transparent colormap
    colors = ['#00000000', '#FFFF00BB', '#FF8000DD', '#FF0000FF']
    cmap = LinearSegmentedColormap.from_list('risk', colors, N=100)
    
    # Rotate the risk grid by ego_yaw
    rotation_degrees = -np.degrees(ego_yaw)  # Negative because image coordinates
    rotated_risk = scipy.ndimage.rotate(risk_grid_2d, rotation_degrees, reshape=True, order=1)
    
    # Get image dimensions
    img_height, img_width = base_img.shape[:2]
    
    # Scale the risk grid to appropriate size in pixels
    # Assume the base image shows roughly 100m x 100m centered on scene
    pixels_per_meter = img_width / 200.0  # Adjust this ratio based on your scene
    
    risk_height_pixels = int(2 * grid_range_long * pixels_per_meter)
    risk_width_pixels = int(2 * grid_range_lat * pixels_per_meter)
    
    # Resize rotated risk grid
    from scipy.ndimage import zoom
    zoom_factor_h = risk_height_pixels / rotated_risk.shape[0]
    zoom_factor_w = risk_width_pixels / rotated_risk.shape[1]
    resized_risk = zoom(rotated_risk, (zoom_factor_h, zoom_factor_w), order=1)
    
    # Center on ego vehicle position (converted to pixels)
    # This is approximate - center of image assumed to be scene center
    center_x_pixel = img_width / 2
    center_y_pixel = img_height / 2
    
    # Calculate extent in pixels
    half_h = resized_risk.shape[0] / 2
    half_w = resized_risk.shape[1] / 2
    
    extent_pixels = [
        center_x_pixel - half_w,
        center_x_pixel + half_w,
        center_y_pixel + half_h,  # Note: y is flipped in images
        center_y_pixel - half_h
    ]
    
    # Plot risk overlay
    im = ax.imshow(
        resized_risk,
        extent=extent_pixels,
        cmap=cmap,
        alpha=0.7,
        vmin=0.0,
        vmax=1.0,
        zorder=10,
        interpolation='bilinear'
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Risk Level', rotation=270, labelpad=15)
    
    
    # === STEP 7: Convert to image ===
    plt.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    
    plt.close(fig)
    
    return img


def plot_risk_grid_aligned(ax, risk_grid, state, ego_idx, grid_range_long=50.0, grid_range_lat=15.0):
    """
    Plot risk grid properly aligned with ego vehicle in world coordinates.
    """
    risk_grid_2d = risk_grid.squeeze()
    
    # Get ego state
    ego_pos = state.sim_trajectory.xy[ego_idx, state.timestep]
    ego_yaw = state.sim_trajectory.yaw[ego_idx, state.timestep]
    
    # Create the extent in ego-centric frame (before rotation)
    # The grid is: longitudinal (forward/back) x lateral (left/right)
    extent = [-grid_range_long, grid_range_long, -grid_range_lat, grid_range_lat]
    
    # Plot the risk grid
    im = ax.imshow(
        risk_grid_2d.T,  # Transpose to get correct orientation
        origin='lower',
        cmap='hot',
        alpha=0.6,
        vmin=0.0,
        vmax=1.0,
        extent=extent,  # In ego frame
        interpolation='bilinear',
        zorder=2  # Below vehicles
    )
    
    # Apply transformation: rotate by ego_yaw and translate to ego_pos
    trans_data = (
        Affine2D()
        .rotate(ego_yaw)  # Rotate to align with ego heading
        .translate(ego_pos[0], ego_pos[1])  # Move to ego position
        + ax.transData
    )
    im.set_transform(trans_data)
    
    return im

# === USAGE IN YOUR VIDEO LOOP ===
def save_risk_video_aligned(states, ego_idx, save_path='risk_video.mp4'):
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    
    frames = []
    
    for timestep, state in enumerate(states):
        # Compute risk
        risk_grid = create_mttc_risk_grid(state, ego_idx)
        risk_grid_2d = risk_grid.squeeze()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get ego info
        ego_pos = state.sim_trajectory.xy[ego_idx, state.timestep]
        ego_yaw = state.sim_trajectory.yaw[ego_idx, state.timestep]
        
        # === 1. PLOT RISK GRID FIRST (with proper alignment) ===
        if risk_grid_2d.max() > 0:
            im = plot_risk_grid_aligned(
                ax, risk_grid, state, ego_idx,
                grid_range_long=50.0,
                grid_range_lat=15.0
            )
            
            # Add colorbar (only once)
            if timestep == 0:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Risk Level', rotation=270, labelpad=20)
        
        # === 2. PLOT VEHICLES ON TOP ===
        for obj_idx in range(state.sim_trajectory.num_objects):
            if not state.object_metadata.is_valid[obj_idx]:
                continue
            
            pos = state.sim_trajectory.xy[obj_idx, state.timestep]
            yaw = state.sim_trajectory.yaw[obj_idx, state.timestep]
            length = state.sim_trajectory.length[obj_idx, state.timestep]
            width = state.sim_trajectory.width[obj_idx, state.timestep]
            
            # Create rectangle centered at vehicle position
            rect = Rectangle(
                xy=(pos[0] - length/2 * np.cos(yaw) + width/2 * np.sin(yaw),
                    pos[1] - length/2 * np.sin(yaw) - width/2 * np.cos(yaw)),
                width=length,
                height=width,
                angle=np.degrees(yaw),
                facecolor='red' if obj_idx == ego_idx else 'cyan',
                edgecolor='black',
                linewidth=2,
                alpha=0.8,
                zorder=5
            )
            ax.add_patch(rect)
            
            # Add arrow showing heading for ego
            if obj_idx == ego_idx:
                arrow_length = length * 0.7
                ax.arrow(
                    pos[0], pos[1],
                    arrow_length * np.cos(yaw),
                    arrow_length * np.sin(yaw),
                    head_width=width * 0.4,
                    head_length=length * 0.3,
                    fc='yellow',
                    ec='black',
                    linewidth=2,
                    zorder=6
                )
        
        # === 3. SET VIEW CENTERED ON EGO ===
        view_range = 80  # meters
        ax.set_xlim(ego_pos[0] - view_range, ego_pos[0] + view_range)
        ax.set_ylim(ego_pos[1] - view_range*0.6, ego_pos[1] + view_range*0.6)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'Risk Grid Visualization - Timestep {timestep}', fontsize=14)
        
        # Capture frame
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(img)
        
        plt.close(fig)
    
    # Save video
    fig_final, ax_final = plt.subplots(figsize=(14, 10))
    im_video = ax_final.imshow(frames[0])
    ax_final.axis('off')
    
    def update(frame_idx):
        im_video.set_array(frames[frame_idx])
        return [im_video]
    
    ani = animation.FuncAnimation(
        fig_final, update, frames=len(frames), interval=100, blit=True
    )
    
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=10, bitrate=1800)
    ani.save(save_path, writer=writer)
    plt.close(fig_final)
    
    print(f"Saved video with {len(frames)} frames to {save_path}")

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Example of how to visualize risk grid during simulation.
    """
    
    # Assuming you have a running simulation:
    # states = []
    # for timestep in range(T):
    #     output = actor.select_action(...)
    #     state = env.step(state, output.action)
    #     states.append(state)
    
    # === Option 1: Single frame visualization ===
    # fig, ax = visualize_risk_grid_overlay(
    #     state=current_state,
    #     ego_idx=av_index,
    #     grid_range_long=50.0,
    #     grid_range_lat=15.0,
    #     show_vehicles=True,
    #     show_trajectories=True,
    #     show_mttc_values=True
    # )
    # plt.show()
    
    # === Option 2: Animation over entire scenario ===
    # anim = animate_scenario_with_risk(
    #     states_list=states,
    #     ego_idx=av_index,
    #     save_path='./risk_visualization.mp4',
    #     grid_range_long=50.0,
    #     grid_range_lat=15.0
    # )
    # plt.show()
    
    pass


if __name__ == "__main__":
    print(__doc__)
    print("\nUsage:")
    print("1. Single frame: visualize_risk_grid_overlay(state, ego_idx)")
    print("2. Animation: animate_scenario_with_risk(states_list, ego_idx)")
    print("\nThe risk grid shows:")
    print("  - Green: Low risk (safe)")
    print("  - Yellow: Medium risk (caution)")
    print("  - Orange/Red: High risk (danger)")
    print("  - Blue vehicle: Ego")
    print("  - Black vehicles: Other agents")
    print("  - Dashed lines: Predicted trajectories")
    print("  - 'R=X.XX' labels: MTTC-based risk values")