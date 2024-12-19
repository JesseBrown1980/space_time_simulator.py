import numpy as np
from typing import Tuple, Union

# Simulate your friend's typing_ and backend modules
# In your real code, replace these with your actual imports.
Number = Union[float, int]
Tensorlike = np.ndarray
bd = np

# Constants (if needed)
# from . import constants as const
# For now we won't define constants as none were used explicitly.

def curl_E(E: Tensorlike) -> Tensorlike:
    """
    Transforms an E-type field into an H-type field by performing a curl operation.
    E is located on edges (integer gridpoints), result is on faces (half-integer grid points).
    """
    curl = bd.zeros(E.shape)

    # Following the original indexing logic
    curl[:, :-1, :, 0] += E[:, 1:, :, 2] - E[:, :-1, :, 2]
    curl[:, :, :-1, 0] -= E[:, :, 1:, 1] - E[:, :, :-1, 1]

    curl[:, :, :-1, 1] += E[:, :, 1:, 0] - E[:, :, :-1, 0]
    curl[:-1, :, :, 1] -= E[1:, :, :, 2] - E[:-1, :, :, 2]

    curl[:-1, :, :, 2] += E[1:, :, :, 1] - E[:-1, :, :, 1]
    curl[:, :-1, :, 2] -= E[:, 1:, :, 0] - E[:, :-1, :, 0]

    return curl

def curl_edge_to_face(v: Tensorlike) -> Tensorlike:
    """
    Compute the curl of a vector field v on edges to get a field on faces.
    """
    return curl_E(v)

def curl_H(H: Tensorlike) -> Tensorlike:
    """
    Transforms an H-type field (faces) into an E-type field (edges) by performing a curl operation.
    """
    curl = bd.zeros(H.shape)

    curl[:, 1:, :, 0] += H[:, 1:, :, 2] - H[:, :-1, :, 2]
    curl[:, :, 1:, 0] -= H[:, :, 1:, 1] - H[:, :, :-1, 1]

    curl[:, :, 1:, 1] += H[:, :, 1:, 0] - H[:, :, :-1, 0]
    curl[1:, :, :, 1] -= H[1:, :, :, 2] - H[:-1, :, :, 2]

    curl[1:, :, :, 2] += H[1:, :, :, 1] - H[:-1, :, :, 1]
    curl[:, 1:, :, 2] -= H[:, 1:, :, 0] - H[:, :-1, :, 0]

    return curl

def curl_face_to_edge(A: Tensorlike) -> Tensorlike:
    """
    Compute the curl of a vector field A on faces, resulting in a field on edges.
    """
    return curl_H(A)

def div(v: Tensorlike) -> Tensorlike:
    """
    Compute the divergence of a vector field v defined on edges.
    The result is a scalar field on faces.
    """
    # Based on your your old code
    div_v = bd.zeros((v.shape[0], v.shape[1], v.shape[2], 1))

    # x-component
    div_v[1:-1, :, :, 0] += (v[1:-1, :, :, 0] - v[:-2, :, :, 0])
    div_v[:-2, :, :, 0] -= (v[1:-1, :, :, 0] - v[2:, :, :, 0])

    # y-component
    div_v[:, 1:-1, :, 0] += (v[:, 1:-1, :, 1] - v[:, :-2, :, 1])
    div_v[:, :-2, :, 0]  -= (v[:, 1:-1, :, 1] - v[:, 2:, :, 1])

    # z-component
    div_v[:, :, 1:-1, 0] += (v[:, :, 1:-1, 2] - v[:, :, :-2, 2])
    div_v[:, :, :-2, 0]  -= (v[:, :, 1:-1, 2] - v[:, :, 2:, 2])

    return div_v

def grad(p: Tensorlike) -> Tensorlike:
    """
    Compute the gradient of a scalar field p located on grid points.
    p has shape (Nx+1, Ny+1, Nz+1, 1).
    Returns (Nx+1, Ny+1, Nz+1, 3).
    """
    grad = bd.zeros((p.shape[0], p.shape[1], p.shape[2], 3))
    
    grad[:-1, :, :, 0] = (p[1:, :, :, 0] - p[:-1, :, :, 0])
    grad[:, :-1, :, 1] = (p[:, 1:, :, 0] - p[:, :-1, :, 0])
    grad[:, :, :-1, 2] = (p[:, :, 1:, 0] - p[:, :, :-1, 0])
    
    return grad

############################################
# Additional functionalities and example
############################################

def laplacian_vector_field(v: Tensorlike) -> Tensorlike:
    """
    Compute the vector Laplacian Δv using the provided curl and div.
    Δv = ∇(∇·v) - ∇×(∇×v)
    To do this correctly, we might need v and p fields on compatible staggered grids.
    Here we assume the same indexing as your friend's code.
    """
    # As an example, let's just apply curl and div in a way similar to the previous code
    # Note: The original code doesn't provide dx, dy, dz, we assume uniform spacing of 1.
    div_v = div(v)              # scalar field
    grad_div_v = grad(div_v)    # vector field
    c = curl_edge_to_face(v)    # curl moves edges->faces
    curl_c = curl_face_to_edge(c) # curl moves faces->edges
    return grad_div_v - curl_c  # Δv = ∇(∇·v) - ∇×(∇×v)

def apply_second_order_operator(v: Tensorlike, k: float = 1.0) -> Tensorlike:
    """
    Apply d/dt = -k Δ to a vector field v on edges.
    Here Δv = laplacian_vector_field(v).
    """
    lap_v = laplacian_vector_field(v)
    return -k * lap_v

# Boundary conditions
def apply_dirichlet_condition(field: np.ndarray, value: float = 0.0):
    field[0, :, :, :] = value
    field[-1, :, :, :] = value
    field[:, 0, :, :] = value
    field[:, -1, :, :] = value
    field[:, :, 0, :] = value
    field[:, :, -1, :] = value

def apply_neumann_condition(field: np.ndarray):
    field[0, :, :, :] = field[1, :, :, :]
    field[-1, :, :, :] = field[-2, :, :, :]
    field[:, 0, :, :] = field[:, 1, :, :]
    field[:, -1, :, :] = field[:, -2, :, :]
    field[:, :, 0, :] = field[:, :, 1, :]
    field[:, :, -1, :] = field[:, :, -2, :]

def apply_periodic_condition(field: np.ndarray):
    field[0, :, :, :] = field[-2, :, :, :] 
    field[-1, :, :, :] = field[1, :, :, :]
    field[:, 0, :, :] = field[:, -2, :, :]
    field[:, -1, :, :] = field[:, 1, :, :]
    field[:, :, 0, :] = field[:, :, -2, :]
    field[:, :, -1, :] = field[:, :, 1, :]

if __name__ == "__main__":
    # Example usage:
    # Suppose Nx, Ny, Nz are the number of cells in each direction
    # The given code suggests fields on staggered grids. Adjust sizes as needed.
    Nx, Ny, Nz = 20, 20, 20
    v_field = bd.random.randn(Nx, Ny, Nz, 3)  # A random vector field
    
    # Choose a boundary type
    boundary_type = 'dirichlet'  # 'dirichlet', 'neumann', 'periodic'
    dt = 0.1
    k = 1.0
    num_timesteps = 5

    for _ in range(num_timesteps):
        dv_dt = apply_second_order_operator(v_field, k)
        v_field = v_field + dt * dv_dt

        # Apply chosen boundary condition
        if boundary_type == 'dirichlet':
            apply_dirichlet_condition(v_field, value=0.0)
        elif boundary_type == 'neumann':
            apply_neumann_condition(v_field)
        elif boundary_type == 'periodic':
            apply_periodic_condition(v_field)

    print("Final field shape:", v_field.shape)
    print("Simulation complete.")
