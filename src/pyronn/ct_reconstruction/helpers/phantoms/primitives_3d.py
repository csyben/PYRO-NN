import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def place_sphere(grid, pos, radius, value=1.0):
    """
    Updates an existing 3D grid by placing a spherical object within it.

    Args:
        grid:   The existing 3D numpy array (meshgrid) to update.
        pos:    Center of the sphere (in [Z, Y, X]).
        radius: Radius of the sphere.
        value:  Value to fill the sphere with.

    Returns:
        None; the function updates the grid in place.
    """

    # Ensure the grid is a numpy array
    grid = np.asarray(grid)

    # Grid shape
    shape = grid.shape

    # Create meshgrid of coords based on the grid's shape
    xx, yy, zz = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]

    # Calculate squared distance to pos
    circle = (xx - pos[2])**2 + (yy - pos[1])**2 + (zz - pos[0])**2

    # Update grid in place where the condition is met
    grid[circle <= radius**2] = value

def place_cube(grid, pos, size, value=1.0):
    """
    Updates an existing 3D grid by placing a cube object within it.

    Args:
        grid:   The existing 3D numpy array (meshgrid) to update.
        pos:    Position of the cube's upper left corner (in [Z, Y, X]).
        size:   Size of the cube (in [Z, Y, X]).
        value:  Value to fill the cube with.

    Returns:
        None; the function updates the grid in place.
    """
    # Ensure pos and size are within the grid bounds
    for i in range(3):
        if pos[i] < 0 or pos[i] + size[i] > grid.shape[i]:
            raise ValueError(f"Cube at position {pos} with size {size} exceeds grid bounds.")

    # Update the grid in place
    grid[pos[0]:pos[0]+size[0], pos[1]:pos[1]+size[1], pos[2]:pos[2]+size[2]] = value
    

def rotate_point_around_z(x, y, z, angle):
    """
    Rotate a point around the Z-axis by a given angle.

    Args:
        x, y, z: Coordinates of the point.
        angle: Rotation angle in radians.

    Returns:
        Rotated coordinates (x', y', z').
    """
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_rotated = x * cos_angle - y * sin_angle
    y_rotated = x * sin_angle + y * cos_angle
    return x_rotated, y_rotated, z


def place_cube_with_rotation(grid, pos, size, value=1.0, angle=None):
    """
    Updates an existing 3D grid by placing a cube object within it, potentially rotated around the Z-axis.

    Args:
        grid:   The existing 3D numpy array (meshgrid) to update.
        pos:    Position of the cube's upper left corner (in [Z, Y, X]).
        size:   Size of the cube (in [Z, Y, X]).
        value:  Value to fill the cube with.
        angle:  Rotation angle in radians around the Z-axis. If None, a random angle is chosen.

    Returns:
        None; the function updates the grid in place.
    """
    if angle is None:
        angle = np.random.uniform(0, 2*np.pi)  # Random angle if not specified

    xx,yy,zz = np.mgrid[0:grid.shape[0], 0:grid.shape[1], 0:grid.shape[2]]

    # Center of the cube
    center = np.array(pos) + np.array(size) / 2

    # Rotate grid points around the Z-axis in the opposite direction
    try:
        xx_rotated, yy_rotated, _ = rotate_point_around_z(xx - center[2], yy - center[1], 0, -angle)
        # Adjust back to grid coordinates
        xx_rotated += center[2]
        yy_rotated += center[1]
         # Check if rotated points are inside the unrotated cube's bounds
        inside_cube = (
            (xx_rotated >= pos[2]) & (xx_rotated <= pos[2] + size[2]) &
            (yy_rotated >= pos[1]) & (yy_rotated <= pos[1] + size[1]) &
            (zz >= pos[0]) & (zz <= pos[0] + size[0])
        )

        # Update grid where the condition is met
        grid[inside_cube] = value
    except ValueError:
        # Calculate start and end indices for each dimension, ensuring they are within the grid bounds
        start_indices = [max(0, p) for p in pos]
        end_indices = [min(pos[i] + size[i], grid.shape[i]) for i in range(3)]

        # Update the grid within the calculated indices
        grid[
            start_indices[0]:end_indices[0],
            start_indices[1]:end_indices[1],
            start_indices[2]:end_indices[2]
        ] = value

    

   

def place_ellipsoid_with_rotation(grid, pos, radii, value=1.0, angle=None):
    """
    Updates an existing 3D grid by placing a randomly rotated ellipsoid object within it.

    Args:
        grid:   The existing 3D numpy array (meshgrid) to update.
        pos:    Center of the ellipsoid (in [Z, Y, X]).
        radii:  Radii of the ellipsoid (in [Z, Y, X]).
        value:  Value to fill the ellipsoid with.
        angle:  Rotation angle in radians around the Z-axis. If None, a random angle is chosen.

    Returns:
        None; the function updates the grid in place.
    """
    if angle is None:
        angle = np.random.uniform(0, 2*np.pi)  # Random angle if not specified

    zz, yy, xx = np.mgrid[0:grid.shape[0], 0:grid.shape[1], 0:grid.shape[2]]

    # Rotate grid points around the Z-axis in the opposite direction
    xx_rotated, yy_rotated, _ = rotate_point_around_z(xx - pos[2], yy - pos[1], 0, -angle)

    # Adjust back to original positions
    xx_rotated += pos[2]
    yy_rotated += pos[1]

    # Check if rotated points are inside the ellipsoid
    inside_ellipsoid = (
        ((xx_rotated - pos[2])**2 / radii[2]**2) +
        ((yy_rotated - pos[1])**2 / radii[1]**2) +
        ((zz - pos[0])**2 / radii[0]**2)
    ) <= 1

    # Update grid where the condition is met
    grid[inside_ellipsoid] = value

def place_cylinder_with_rotation(grid, pos, height, radius, value=1.0, angle=None):
    """
    Updates an existing 3D grid by placing a randomly rotated cylinder object within it.
    Args:
        grid: The existing 3D numpy array (meshgrid) to update.
        pos: Center of the base of the cylinder (in [Z, Y, X]).
        height: Height of the cylinder along the Z-axis.
        radius: Radius of the cylinder in the XY plane.
        value: Value to fill the cylinder with.
        angle: Rotation angle in radians around the Z-axis. If None, a random angle is chosen.
    Returns:
        None; the function updates the grid in place.
    """
    if angle is None:
        angle = np.random.uniform(0, 2*np.pi)  # Random angle if not specified

    # Create meshgrid
    zz, yy, xx = np.mgrid[0:grid.shape[0], 0:grid.shape[1], 0:grid.shape[2]]

    # Rotate grid points around the Z-axis in the opposite direction to simulate cylinder rotation
    xx_rotated, yy_rotated, _ = rotate_point_around_z(xx - pos[2], yy - pos[1], zz, -angle)

    # Adjust back to original positions
    xx_rotated += pos[2]
    yy_rotated += pos[1]

    # Check if rotated points are within the cylinder
    inside_cylinder = ((xx_rotated - pos[2])**2 + (yy_rotated - pos[1])**2 <= radius**2) & \
                      (zz >= pos[0]) & (zz <= pos[0] + height)

    # Update grid where the condition is met
    grid[inside_cylinder] = value

def place_pyramid(grid, base_center, base_size, height, value=1.0):
    """
    Places an axis-aligned pyramid into a 3D grid.

    Args:
        grid: 3D numpy array representing the grid.
        base_center: Center of the pyramid's base in the grid (z, y, x).
        base_size: Side length of the pyramid's square base.
        height: Height of the pyramid.
        value: Value to fill the pyramid with.

    Returns:
        None; the grid is modified in place.
    """
    zz, yy, xx = np.mgrid[0:grid.shape[0], 0:grid.shape[1], 0:grid.shape[2]]
    
    # Calculate distances from the base center in the XY plane
    dx = np.abs(xx - base_center[2])
    dy = np.abs(yy - base_center[1])

    # Calculate the maximum allowable distance in the XY plane at each Z level
    max_dist = base_size / 2 * (1 - zz / height)

    # Determine points inside the pyramid
    inside_pyramid = (dx <= max_dist) & (dy <= max_dist) & (zz <= height)

    # Update the grid
    grid[inside_pyramid] = value

def visualize_grid(grid):
    """
    Visualizes a 3D grid using matplotlib, plotting points where the grid value is non-zero.

    Args:
        grid: The 3D numpy array to visualize.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the coordinates of points where the grid value is non-zero
    z, y, x = grid.nonzero()

    # Use scatter plot for these points
    ax.scatter(x, y, z, c='red', marker='o')

    # Set labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Grid Visualization')

    # plt.show()
    plt.savefig(f'random_object.png', dpi=150, transparent=False, bbox_inches='tight')

def generate_3D_primitives(volume_shape, number_of_primitives):
    """
    Generates a 3D phantom composed of a specified number of random geometric primitives. The primitives are added
    to the phantom with random positions, orientations, sizes, and intensities. The method returns both
    the phantom and its sinogram as PyTorch tensors.

    Parameters:
    - volume_shape: shape of the phantom
    - number_of_primitives (int, optional): The number of geometric primitives to include in the phantom.
    Defaults to 6.

    Returns:
    - numpy.array: The 3D phantom.
    """
    grid = np.zeros(volume_shape)

    for i in range(number_of_primitives):
        object_type = random.choice(["ellipsoid", "sphere", "cube", "pyramid", "cylinder", "rectangle"])
        pos = np.random.randint(0, volume_shape[0], 3)
        intensitiy_value = np.random.uniform(0.4, 1.0, 1)
        print(f"{i}th Random choice was {object_type}, placed at {pos} with intensity {intensitiy_value}.")

        if object_type == "ellipsoid":
            ellipsoid_radii = np.random.randint(1, int(volume_shape[0] / 5),
                                                3)  # Radii along Z, Y, X axes
            place_ellipsoid_with_rotation(grid, pos, ellipsoid_radii, value=intensitiy_value)
        elif object_type == "sphere":
            radius = np.random.randint(1, int(volume_shape[0] / 5), 1)  # Radius
            place_sphere(grid, pos, radius, value=intensitiy_value)
        elif object_type == 'rectangle':
            cube_size = np.random.randint(1, int(volume_shape[0] / 5), 3)  # Size of the cube
            place_cube_with_rotation(grid, pos, cube_size, value=intensitiy_value)
        elif object_type == 'cube':
            cube_size = np.random.randint(1, int(volume_shape[0] / 5), 1)  # Size of the cube
            place_cube_with_rotation(grid, pos, (cube_size[0], cube_size[0], cube_size[0]), value=intensitiy_value)
        elif object_type == 'pyramid':
            base_size = np.random.randint(1, int(volume_shape[0] / 5), 1)  # Length of the base's side
            height = np.random.randint(1, int(volume_shape[0] / 5), 1)  # Height of the pyramid
            # Place the pyramid in the grid
            place_pyramid(grid, pos, base_size, height, intensitiy_value)
        else:  # 'cylinder'
            cylinder_height = np.random.randint(1, int(volume_shape[0] / 5), 1)
            cylinder_radius = np.random.randint(1, int(volume_shape[0] / 5), 1)
            place_cylinder_with_rotation(grid, pos, cylinder_height, cylinder_radius, value=intensitiy_value)
    return grid

# Example usage
if __name__ == "__main__":
    # Initialize a 3D grid with zeros
    grid_shape = (100, 100, 100)  # Define the size of your grid
    grid = np.zeros(grid_shape)

    # Place the first sphere
    sphere1_pos = (50, 50, 50)  # Center of the first sphere
    sphere1_radius = 10  # Radius of the first sphere
    place_sphere(grid, sphere1_pos, sphere1_radius, value=0.4)

    # Place the cube
    cube_pos = (20, 20, 20)  # Upper left corner of the cube
    cube_size = (10, 10, 10)  # Size of the cube
    place_cube_with_rotation(grid, cube_pos, cube_size, value=0.2)

    # Place an ellipsoid
    # ellipsoid_pos = (5, 50, 5)  # Center of the ellipsoid
    # ellipsoid_radii = (20, 10, 30)  # Radii along Z, Y, X axes
    # place_ellipsoid_with_rotation(grid, ellipsoid_pos, ellipsoid_radii, value=1.0)

    # Visualize the grid
    # visualize_grid(grid)
