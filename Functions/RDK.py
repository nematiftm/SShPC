import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def update_points_within_circle(x_positions, y_positions, circle_center_x, circle_center_y, circle_radius):
    angle = np.arcsin((y_positions - circle_center_y) / circle_radius)
    x = circle_center_x + circle_radius * np.cos(angle)
    larger_indices = x_positions > x
    x_positions[larger_indices] = -1 * x[larger_indices]
    smaller_indices = x_positions < (-1 * x)
    x_positions[smaller_indices] = x[smaller_indices]
    
    angle = np.arccos((x_positions - circle_center_x) / circle_radius)
    y = circle_center_y + circle_radius * np.sin(angle)
    out_of_bounds = (y_positions > y) | (y_positions < (-1 * y))
    y_positions[out_of_bounds] = np.sign(y_positions[out_of_bounds]) * y[out_of_bounds]

    return x_positions, y_positions

def rdk_exp(num_dots=100, noise_prop=0.2, x_velocity_type="constant", x_velocity_value=0.01, y_velocity=0.01, fr=50, dot_size=3, radius=0.5, name='random_dots.mp4', generate_animation=False):
    # Ensure the Animations directory exists
    animation_dir = "Animations"
    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)
    
    # Update file name with the directory path
    name = os.path.join(animation_dir, name)

    direction = np.random.choice([1, -1])

    # Generate the x and y positions in a circle
    # Generate a random angle between 0 and 2*pi
    circle_center_x = 0
    circle_center_y = 0
    angle = np.random.uniform(0, 2 * np.pi, num_dots)
    
    # Generate a random distance from the center between 0 and radius
    distance = np.sqrt(np.random.uniform(0, radius**2, num_dots))
    
    # Calculate the coordinates of the random dot
    x_positions = circle_center_x + distance * np.cos(angle)
    y_positions = circle_center_y + distance * np.sin(angle)

    if x_velocity_type == "constant":
        x_velocities = np.full(num_dots, x_velocity_value)  # Constant x-velocity
    elif x_velocity_type == "random":
        x_velocities = np.random.uniform(0.01, 0.03, num_dots)  # Random x-velocities
    else:
        x_velocities = np.full(num_dots, x_velocity_value)  # Specific x-velocity value

    # Calculate the number of dots with random vertical movement based on SNR
    num_random_dots = int(noise_prop * num_dots)
    num_left_dots = int(num_random_dots / 2)
    num_right_dots = num_random_dots - num_left_dots

    # Generate random velocities for the dots with random movement
    y_velocities_left = np.random.uniform(-1 * y_velocity, y_velocity, num_left_dots)
    y_velocities_right = np.random.uniform(-1 * y_velocity, y_velocity, num_right_dots)

    x_velocities_left = np.random.uniform(-1 * (y_velocity / 4), (y_velocity / 4), num_left_dots)
    x_velocities_right = np.random.uniform(-1 * (y_velocity / 4), (y_velocity / 4), num_right_dots)

    # Create an empty matrix to store the dot information for each frame
    dot_matrix = np.zeros((fr, num_dots, 5))  # Change 50 to the desired number of frames

    # Define the update function for the animation
    def update(frame):
        nonlocal x_positions, y_positions, x_velocities, dot_matrix

        # Generate noise for vertical movement of dots
        noise_y = np.zeros(num_dots)
        noise_x = np.ones(num_dots) * direction * x_velocities

        # Select random dots for noise
        random_indices_left = np.random.choice(num_dots, num_left_dots, replace=False)
        random_indices_right = np.random.choice(num_dots, num_right_dots, replace=False)

        # Generate random vertical movement for selected dots
        noise_y[random_indices_left] = y_velocities_left
        noise_y[random_indices_right] = y_velocities_right
        noise_x[random_indices_left] = x_velocities_left
        noise_x[random_indices_right] = x_velocities_right

        ydirection = np.random.choice([1, -1])
        # Update the vertical positions of dots with noise
        y_positions += ydirection * noise_y
        x_positions += noise_x

        x_positions, y_positions = update_points_within_circle(x_positions, y_positions, circle_center_x, circle_center_y, radius)

        if generate_animation:
            # Update the scatter plot with the new positions
            dots.set_offsets(np.column_stack((x_positions, y_positions)))

        # Calculate the total velocity for each dot
        v_total = np.sqrt(noise_x**2 + noise_y**2)

        # Update the dot_matrix with the dot information for the current frame
        dot_matrix[frame, :, 0] = x_positions
        dot_matrix[frame, :, 1] = y_positions
        dot_matrix[frame, :, 2] = noise_x
        dot_matrix[frame, :, 3] = noise_y
        dot_matrix[frame, :, 4] = v_total

        if generate_animation:
            return dots,

    if generate_animation:
        # Set up the figure and axis
        fig, ax = plt.subplots(facecolor='black')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_axis_off()
        # Create a scatter plot with the initial positions and smaller dots
        dots = ax.scatter(x_positions, y_positions, color='white', s=dot_size)
        # Set the animation interval and repeat
        ani = animation.FuncAnimation(fig, update, frames=fr, interval=50, blit=True)

        # Save the animation as an MP4 file in the Animations folder
        ani.save(name, writer='ffmpeg')
        plt.close()
    else:
        for frame in range(fr):
            update(frame)  # Call the update function and store the returned value

    if direction == 1:
        direct = "right"
    else:
        direct = "left"

    if generate_animation:
        print("Direction is " + direct)
    
    return dot_matrix
