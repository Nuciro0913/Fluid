import numpy as np
import pytest
from solver import FluidSolver

def test_solver_init():
    nx, ny = 20, 10
    solver = FluidSolver(nx=nx, ny=ny)
    assert solver.u.shape == (nx + 1, ny)
    assert solver.v.shape == (nx, ny + 1)
    assert solver.p.shape == (nx, ny)

def test_solver_step():
    nx, ny = 20, 10
    solver = FluidSolver(nx=nx, ny=ny)
    # Just check if it runs without error
    solver.step(inlet_velocity=1.0)
    assert not np.isnan(solver.u).any()
    assert not np.isnan(solver.v).any()

def test_obstacle_setting():
    nx, ny = 20, 10
    solver = FluidSolver(nx=nx, ny=ny)
    mask = np.ones((nx, ny), dtype=bool)
    mask[5:10, 3:7] = False # Add obstacle
    solver.set_obstacle(mask)
    assert not solver.mask[5, 3]
    # Check if velocity is zeroed at obstacle (rough check)
    assert solver.u[5, 3] == 0
