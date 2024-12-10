import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import MinMaxScaler
from Functions.functions import *

def johansson_exp(initials, num_obj, T, dt, vel_params, name_obj, dot_colors, 
                       anim_name, plot=False, scale_data=False, add_noise=False,
                       noise_std=0.1, generate_animation=False):
    def velocity(time, params):
        vx, vy = [
            params[axis]['A'] * np.sin(2 * np.pi * params[axis]['f'] * time + params[axis]['phase'])
            if params[axis]['type'] == 'sin' else params[axis]['val']
            for axis in ['x', 'y']
        ]
        return vx, vy

    # Ensure the Animations directory exists
    animation_dir = "Animations"
    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)

    # Update animation name to include the Animations directory path
    anim_name = os.path.join(animation_dir, anim_name)

    # Loop through each time step and calculate the location
    num_steps = int(T / dt)
    loc_list = np.zeros((num_steps + 1, num_obj, 6))
    loc_list[0] = initials
    time_array = np.arange(num_steps) * dt

    for i in range(num_steps):
        vels = np.array([velocity(time_array[i], vel_params[j]) for j in range(num_obj)])

        if add_noise:
            # Add noise to velocities
            noise = np.random.normal(loc=0, scale=noise_std, size=vels.shape)
            vels += noise

        prev_ = loc_list[i, :, :]
        prev_[:, 2:4] = vels
        prev_[:, 5] = np.degrees(np.arctan2(vels[:, 1], vels[:, 0]))
        prev_[:, 4] = (np.sqrt(vels[:, 0] ** 2 + vels[:, 1] ** 2)) * np.sign(vels[:, 0])
        prev_[:, 0:2] += prev_[:, 2:4] * dt
        loc_list[i + 1, :, :] = prev_

    dist = distance_dots(loc_list[:, :, 0:2], threshold=0.4)
    data = np.concatenate((loc_list, dist), axis=2)

    if scale_data:
        # Get the shape of the input data
        original_shape = data.shape

        # Reshape the data to have shape (num_samples, num_features)
        reshaped_data = data.reshape(original_shape[0], -1)

        # Initialize MinMaxScaler
        scaler = MinMaxScaler()

        # Fit and transform the scaler on the reshaped data
        scaled_data = scaler.fit_transform(reshaped_data)

        # Reshape the scaled data back to the original shape
        data = scaled_data.reshape(original_shape)

    val_name = np.array([
        'Trajectory of X', 'Trajectory of Y', 'V in X Direction', 'V in Y Direction',
        'V amplitude', 'Theta between Vy and Vx (Degrees)', 'Distance (outer-up from others)',
        'Distance (middle from others)', 'Distance (outer-down from others)', 'Neighbours (outer-up from others)',
        'Neighbours (middle from others)', 'Neighbours (outer-down from others)'
    ])

    # Plot the x and y values over time if plot=True
    if plot:
        fig, axs = plt.subplots(nrows=int(np.shape(data)[2] / 2), ncols=2, sharex=True, figsize=(8, 12))
        fig.subplots_adjust(hspace=0.5)
        for i in range(np.shape(data)[2]):
            axs[np.mod(i, int(np.shape(data)[2] / 2)), int(i / (np.shape(data)[2] / 2))].plot(time_array,
                                                                                                 data[1:, :, i])
            axs[np.mod(i, int(np.shape(data)[2] / 2)), int(i / (np.shape(data)[2] / 2))].set_title(val_name[i])
            axs[np.mod(i, int(np.shape(data)[2] / 2)), int(i / (np.shape(data)[2] / 2))].tick_params(axis='both',
                                                                                                        which='major',
                                                                                                        labelsize=8)
            axs[np.mod(i, int(np.shape(data)[2] / 2)), int(i / (np.shape(data)[2] / 2))].yaxis.set_major_locator(
                MaxNLocator(nbins=5))
            axs[np.mod(i, int(np.shape(data)[2] / 2)), int(i / (np.shape(data)[2] / 2))].set_ylim(auto=True)
            axs[np.mod(i, int(np.shape(data)[2] / 2)), int(i / (np.shape(data)[2] / 2))].grid(True)
            axs[np.mod(i, int(np.shape(data)[2] / 2)), int(i / (np.shape(data)[2] / 2))].legend(name_obj,
                                                                                                  loc='upper left',
                                                                                                  fontsize=8)

        axs[-1, 0].set_xlabel('Time (Seconds)')
        axs[-1, 1].set_xlabel('Time (Seconds)')
        plt.show()

    if generate_animation:
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('white')  # Set the background color to white
        x_min = np.min(data[:, :, 0])-0.1  # Minimum x-coordinate
        x_max = np.max(data[:, :, 0])+0.1  # Maximum x-coordinate
        y_min = np.min(data[:, :, 1])-0.1  # Minimum y-coordinate
        y_max = np.max(data[:, :, 1])+0.1  # Maximum y-coordinate
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axis('off')  # Hide the axes

        # Create three scatter plots for the balls, each with a different color
        dots = [ax.plot([], [], 'wo', markersize=10, color=color)[0] for color in dot_colors]
        
        def animate(i):
            for j in range(num_obj):
                x = data[i, j, 0]
                y = data[i, j, 1]
                dots[j].set_data(x, y)
            return dots

        anim = FuncAnimation(fig, animate, frames=num_steps, interval=30)

        # Save the animation in the Animations folder
        anim.save(anim_name, writer='ffmpeg')
        plt.close()

    return data[1:, ...]
