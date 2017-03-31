# Functions for stimulus-aligning the data; organizing
# data by trial versus sequentially and reshaping between
# these two forms; etc.

import numpy as np


def align_by_stimulus(movie, stimulus_vector, time_before, time_after, reward_vector):
    n_frames, n_pixels = movie.shape
    f = {}
    f['stimulus_times'] = np.where(stimulus_vector != 0)[0]
    f['stimulus_contrast'] = stimulus_vector[f['stimulus_times']]
    n_trials = f['stimulus_times'].size
    trial_length = time_after + time_before
    f['start_times'] = np.maximum(0, f['stimulus_times'] - time_before)
    f['end_times'] = np.minimum(n_frames, f['stimulus_times'] + time_after)
    f['data'] = np.empty((n_trials, trial_length, n_pixels))
    for i in range(n_trials):
        f['data'][i] = movie[f['start_times'][i]:f['end_times'][i],:]
    f['reward_times'] = np.where(reward_vector != 0)[0]
    f['was_reward'] = np.zeros(n_trials)
    for i in range(f['reward_times'].size):
        f['was_reward'] = np.logical_or(f['was_reward'], np.logical_and(f['reward_times'][i] > f['stimulus_times'], f['reward_times'][i] < f['end_times']))
    return f

# Takes a movie that is organized by trials and returns a movie organized sequentially
# (3-dim to 2-dim). Left and right truncation values can be used if you don't want to
# use the full length of the trial.
def reshape_trial_to_sequence(movie, left_truncation=0, right_truncation=0):
    n_trials, n_timesteps, n_pixels = movie.shape
    n_samples = n_trials*(n_timesteps - left_truncation - right_truncation)
    start_idx = left_truncation
    if right_truncation == 0:
        return movie[:,left_truncation:,:].reshape((n_samples, n_pixels))
    else:
        return movie[:,left_truncation:-right_truncation,:].reshape((n_samples, n_pixels))


def reshape_sequence_to_trial(movie, n_trials):
    n_samples, n_pixels = movie.shape
    return movie.reshape((n_trials, n_samples/n_trials, n_pixels))