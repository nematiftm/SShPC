import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def duncker_exp(length, horizontal_speed_ball1, horizontal_speed_ball2, 
                        start_loc_ball2, time_interval,
                        generate_animation=False, save_data=False, animation_name='animation.mp4'):

    # Ensure the Animations directory exists
    animation_dir = "Animations"
    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)

    # Update the animation name to include the directory path
    animation_name = os.path.join(animation_dir, animation_name)

    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    len = length
    ax.set_xlim(-1*len, len)  # Adjust the limits based on your preference
    ax.set_ylim(-15, 15)
    ax.axis('off')  # Turn off the plot axis

    # Ball properties
    ball1, = ax.plot([], [], 'o', markersize=10, color='tab:blue')
    ball2, = ax.plot([], [], 'o', markersize=10, color='tab:orange')

    # Parameters for circular motion
    center_x, center_y = -10, 0  # Center of the circular path
    radius = 10  # Radius of the circular path
    angular_speed = 2.0  # Angular speed of rotation (increased to 2 for 2 rotations)

    # Set the starting location of ball2
    ball2_x, ball2_y = start_loc_ball2, center_y

    # Calculate the total number of frames based on the horizontal movement and time_interval for both balls
    total_number_of_frames_ball1 = int(len / horizontal_speed_ball1 / time_interval)
    total_number_of_frames_ball2 = int(len / horizontal_speed_ball1 / time_interval)
    total_number_of_frames = max(total_number_of_frames_ball1, total_number_of_frames_ball2)

    # Create an empty NumPy array to store data for both balls
    data = np.empty((total_number_of_frames, 2, 5))

    def init():
        ball1.set_data([], [])
        ball2.set_data([], [])
        return ball1, ball2

    def update(frame):
        # Calculate ball1's position for circular motion
        angle = angular_speed * frame * time_interval
        ball1_x = center_x + radius * np.cos(angle) + horizontal_speed_ball1 * frame * time_interval
        ball1_y = center_y + radius * np.sin(angle)

        # Check if ball1's x-coordinate reaches the x-coordinate limit
        if ball1_x >= len:
            ball1_x = len

        # Update ball1's position
        ball1.set_data(ball1_x, ball1_y)

        # Calculate ball2's position for horizontal movement only
        ball2_x = start_loc_ball2 + horizontal_speed_ball2 * frame * time_interval

        # Check if ball2's x-coordinate reaches the x-coordinate limit
        if ball2_x >= len:
            ball2_x = len

        # Update ball2's position
        ball2.set_data(ball2_x, ball2_y)
        
        # Store ball1 data into the data array
        if frame == 0:
            data[frame, 0, 0] = center_x
            data[frame, 0, 1] = center_y
            data[frame, 0, 2] = 0
            data[frame, 0, 3] = 0
            data[frame, 0, 4] = 0

            # Store ball2 data into the data array
            data[frame, 1, 0] = start_loc_ball2
            data[frame, 1, 1] = center_y
            data[frame, 1, 2] = 0
            data[frame, 1, 3] = 0
            data[frame, 1, 4] = 0

        else:
            data[frame, 0, 0] = ball1_x
            data[frame, 0, 1] = ball1_y
            data[frame, 0, 2] = (ball1_x - data[frame-1, 0, 0]) / time_interval 
            data[frame, 0, 3] = (ball1_y - data[frame-1, 0, 1]) / time_interval
            V_total_ball1 = np.sqrt(data[frame, 0, 2]**2 + data[frame, 0, 3]**2) 
            data[frame, 0, 4] = V_total_ball1 

            # Store ball2 data into the data array
            data[frame, 1, 0] = ball2_x
            data[frame, 1, 1] = ball2_y
            data[frame, 1, 2] = horizontal_speed_ball2
            data[frame, 1, 3] = 0
            data[frame, 1, 4] = horizontal_speed_ball2

        return ball1, ball2

    # Create the animation
    ani = FuncAnimation(fig, update, frames=total_number_of_frames, init_func=init, blit=True)
    plt.close()
    
    # Save the animation as an mp4 file
    if generate_animation:
        ani.save(animation_name, fps=30, extra_args=['-vcodec', 'libx264'])
        plt.close()

    if save_data:
        return data

    return None
