import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from solver import FluidSolver
import time

st.set_page_config(page_title="2D CFD Pipe Flow", layout="wide")

st.title("2D FVM CFD Solver - Pipe Flow with Obstacles")

# --- Sidebar: Parameters ---
st.sidebar.header("Simulation Parameters")
nx = st.sidebar.slider("Grid X", 20, 200, 100)
ny = nx // 2
re = st.sidebar.slider("Reynolds Number (Re)", 100, 5000, 2000)
inlet_v = st.sidebar.slider("Inlet Velocity", 0.1, 2.0, 1.0)
dt = st.sidebar.slider("Time Step (dt)", 0.001, 0.05, 0.01)

st.sidebar.header("Obstacle Controls")
obs_x = st.sidebar.slider("Obstacle X Position", 0.1, 0.9, 0.3)
obs_y = st.sidebar.slider("Obstacle Y Position", 0.1, 0.9, 0.5)
obs_r = st.sidebar.slider("Obstacle Radius", 0.02, 0.2, 0.1)

# --- Initialize Solver ---
if "solver" not in st.session_state or st.sidebar.button("Reset Simulation"):
    st.session_state.solver = FluidSolver(nx=nx, ny=ny, re=re, dt=dt)
    st.session_state.steps = 0

solver = st.session_state.solver
# Update parameters in case they changed
solver.re = re
solver.nu = 1.0 / re
solver.dt = dt

# --- Setup Obstacle ---
# lx=2.0, ly=1.0 by default
X, Y = np.meshgrid(np.linspace(0, 2.0, nx), np.linspace(0, 1.0, ny), indexing='ij')
dist = np.sqrt((X - obs_x * 2.0)**2 + (Y - obs_y * 1.0)**2)
mask = dist > obs_r
solver.set_obstacle(mask)

# --- Computation Loop ---
col1, col2 = st.columns([3, 1])

with col2:
    st.write(f"Steps: {st.session_state.steps}")
    run = st.toggle("Run Simulation", value=False)
    num_steps_per_frame = st.number_input("Steps per frame", 1, 100, 5)

placeholder = col1.empty()

def plot_flow(solver):
    fig, ax = plt.subplots(figsize=(10, 5))
    mag = solver.get_velocity_magnitude()
    
    # Plot magnitude
    im = ax.imshow(mag.T, origin='lower', extent=[0, 2, 0, 1], cmap='viridis', aspect='auto')
    
    # Plot streamlines or vectors
    x = np.linspace(0, 2, nx)
    y = np.linspace(0, 1, ny)
    u_c = 0.5 * (solver.u[:-1, :] + solver.u[1:, :])
    v_c = 0.5 * (solver.v[:, :-1] + solver.v[:, 1:])
    
    # Skip points for cleaner streamline
    skip = 5
    # For streamline, x and y must be 1D, u and v (ny, nx)
    ax.streamplot(x, y, u_c.T, v_c.T, color='white', linewidth=0.5, density=0.8)
    
    # Overlay obstacle
    circle = plt.Circle((obs_x * 2.0, obs_y * 1.0), obs_r, color='red', alpha=0.5)
    ax.add_patch(circle)
    
    ax.set_title(f"Velocity Magnitude and Streamlines (Re={re})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(im, ax=ax, label="Velocity")
    return fig

# Main loop
if run:
    while run:
        for _ in range(num_steps_per_frame):
            solver.step(inlet_velocity=inlet_v)
            st.session_state.steps += 1
        
        with placeholder:
            st.pyplot(plot_flow(solver))
        
        time.sleep(0.01)
        # Check if toggle changed (Streamlit might rerun automatically on interaction)
        # But in a while loop we need to be careful.
        # Streamlit-native way: st.rerun() but it resets the loop.
        # To avoid blocking, usually we just do one update per rerun.
        # However, for CFD we want multiple steps.
        st.rerun()
else:
    with placeholder:
        st.pyplot(plot_flow(solver))

st.info("Adjust the sliders to move the obstacle or change Reynolds number. Toggle 'Run Simulation' to start.")
