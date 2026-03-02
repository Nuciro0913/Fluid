import numpy as np
from scipy.ndimage import map_coordinates

class FluidSolver:
    def __init__(self, nx=100, ny=50, lx=2.0, ly=1.0, re=2000.0, dt=0.01):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / nx
        self.dy = ly / ny
        self.re = re
        self.nu = 1.0 / re
        self.dt = dt

        # Staggered grid
        # u is at (nx+1, ny)
        self.u = np.zeros((nx + 1, ny))
        # v is at (nx, ny+1)
        self.v = np.zeros((nx, ny + 1))
        # p is at (nx, ny)
        self.p = np.zeros((nx, ny))
        # Obstacle mask (nx, ny)
        self.mask = np.ones((nx, ny), dtype=bool)

    def set_obstacle(self, obstacle_mask):
        """
        obstacle_mask: boolean array (nx, ny), True where fluid, False where obstacle
        """
        self.mask = obstacle_mask
        # Zero out velocity inside obstacles
        # For simplicity, we zero out velocity at faces adjacent to obstacle cells
        for i in range(self.nx):
            for j in range(self.ny):
                if not self.mask[i, j]:
                    self.u[i, j] = 0
                    self.u[i + 1, j] = 0
                    self.v[i, j] = 0
                    self.v[i, j + 1] = 0

    def step(self, inlet_velocity=1.0):
        # 1. Advection (Semi-Lagrangian)
        u_new = self.advect(self.u, "u")
        v_new = self.advect(self.v, "v")

        # 2. Diffusion (Explicit for simplicity)
        u_new = self.diffuse(u_new, "u")
        v_new = self.diffuse(v_new, "v")

        # 3. Apply Boundary Conditions (Inlet/Outlet/Walls)
        u_new, v_new = self.apply_bc(u_new, v_new, inlet_velocity)

        # 4. Pressure Projection
        # Compute divergence
        div = self.compute_divergence(u_new, v_new)
        
        # Solve Poisson equation for p
        self.p = self.solve_poisson(div)
        
        # Correct velocity
        self.u, self.v = self.project(u_new, v_new, self.p)

    def advect(self, field, field_type):
        nx, ny = field.shape
        coords = np.indices((nx, ny)).astype(float)
        
        if field_type == "u":
            # Field u is at (i*dx, (j+0.5)*dy)
            # Interpolate velocity at u positions
            u_at_u = field
            v_at_u = 0.25 * (self.v[:-1, :-1] + self.v[:-1, 1:] + self.v[1:, :-1] + self.v[1:, 1:])
            # For u_at_u, we need to handle boundaries
            v_at_u_padded = np.zeros_like(u_at_u)
            v_at_u_padded[1:-1, :] = v_at_u
            
            back_x = coords[0] - u_at_u * self.dt / self.dx
            back_y = coords[1] - v_at_u_padded * self.dt / self.dy
            
        elif field_type == "v":
            # Field v is at ((i+0.5)*dx, j*dy)
            u_at_v = 0.25 * (self.u[:-1, :-1] + self.u[1:, :-1] + self.u[:-1, 1:] + self.u[1:, 1:])
            v_at_v = field
            u_at_v_padded = np.zeros_like(v_at_v)
            u_at_v_padded[:, 1:-1] = u_at_v
            
            back_x = coords[0] - u_at_v_padded * self.dt / self.dx
            back_y = coords[1] - v_at_v * self.dt / self.dy
            
        return map_coordinates(field, [back_x, back_y], order=1, mode='nearest')

    def diffuse(self, field, field_type):
        # Explicit diffusion: f_new = f + dt * nu * laplacian(f)
        # Laplacian calculation
        lap = np.zeros_like(field)
        dx2 = self.dx**2
        dy2 = self.dy**2
        
        lap[1:-1, 1:-1] = (
            (field[2:, 1:-1] - 2 * field[1:-1, 1:-1] + field[:-2, 1:-1]) / dx2 +
            (field[1:-1, 2:] - 2 * field[1:-1, 1:-1] + field[1:-1, :-2]) / dy2
        )
        return field + self.dt * self.nu * lap

    def apply_bc(self, u, v, inlet_v):
        # Inlet (Left)
        u[0, :] = inlet_v
        v[0, :] = 0
        
        # Outlet (Right) - Zero gradient
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]
        
        # Walls (Top/Bottom) - No slip or Free slip?
        # Let's do No-slip for pipe flow
        u[:, 0] = 0
        u[:, -1] = 0
        v[:, 0] = 0
        v[:, -1] = 0
        
        # Obstacles
        # For simplicity, just zero out velocity on obstacle faces
        # This is very rough but fits within a script
        for i in range(self.nx):
            for j in range(self.ny):
                if not self.mask[i, j]:
                    u[i, j] = 0
                    u[i+1, j] = 0
                    v[i, j] = 0
                    v[i, j+1] = 0
        
        return u, v

    def compute_divergence(self, u, v):
        div = (u[1:, :] - u[:-1, :]) / self.dx + (v[:, 1:] - v[:, :-1]) / self.dy
        return div

    def solve_poisson(self, div, max_iters=50):
        # Jacobi iteration
        p = self.p.copy()
        dx2 = self.dx**2
        dy2 = self.dy**2
        coeff = 0.5 * dx2 * dy2 / (dx2 + dy2)
        
        for _ in range(max_iters):
            p_old = p.copy()
            # Boundary conditions for pressure (Neumann: dp/dn = 0)
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]
            
            # Interior
            p[1:-1, 1:-1] = coeff * (
                (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) / dx2 +
                (p_old[1:-1, 2:] + p_old[1:-1, :-2]) / dy2 -
                div[1:-1, 1:-1] / self.dt
            )
            
            # Obstacle pressure boundary? 
            # For now, just let it be. Correct way is more complex.
            
        return p

    def project(self, u, v, p):
        u[1:-1, :] -= self.dt * (p[1:, :] - p[:-1, :]) / self.dx
        v[:, 1:-1] -= self.dt * (p[:, 1:] - p[:, :-1]) / self.dy
        
        # Re-apply BCs
        # (Included in next step essentially)
        return u, v

    def get_velocity_magnitude(self):
        # Interpolate u, v to cell centers
        u_center = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        v_center = 0.5 * (self.v[:, :-1] + self.v[:, 1:])
        return np.sqrt(u_center**2 + v_center**2)
