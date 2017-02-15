# Functions for stimulus-aligning the data; organizing
# data by trial versus sequentially and reshaping between
# these two forms; etc.

import numpy as np


def align_by_stimulus(movie, stimulus_vector, time_before, time_after):
    n_pixels, n_frames = movie.shape
    stimulus_times = np.where(stimulus_vector != 0)[0]
    n_trials = stimulus_times.size
    trial_length = time_after - time_before
    start_times = np.maximum(0, stimulus_times - time_before)
    end_times = np.minimum(n_frames, stimulus_times + time_after)
    new_movie = np.empty((n_trials, trial_length, n_pixels))
    for i in range(n_trials):
        new_movie[i] = movie[:,start_times[i]:end_times[i]].T
    return new_movie