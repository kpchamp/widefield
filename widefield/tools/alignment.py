# Functions for stimulus-aligning the data; organizing
# data by trial versus sequentially and reshaping between
# these two forms; etc.

import numpy as np


def align_by_stimulus(movie, stimulus_vector, time_before, time_after):
    n_frames, n_pixels = movie.shape
    stimulus_times = np.where(stimulus_vector != 0)[0]
    n_trials = stimulus_times.size
    trial_length = time_after + time_before
    start_times = np.maximum(0, stimulus_times - time_before)
    end_times = np.minimum(n_frames, stimulus_times + time_after)
    new_movie = np.empty((n_trials, trial_length, n_pixels))
    for i in range(n_trials):
        new_movie[i] = movie[start_times[i]:end_times[i],:]
    return new_movie

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
    n_pixels, n_samples = movie.shape
    return movie.reshape((n_trials, n_samples/n_trials, n_pixels))