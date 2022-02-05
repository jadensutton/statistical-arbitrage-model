import copy
import math

import numpy as np
import matplotlib.pyplot as plt

from pykalman import KalmanFilter
from statistics import mean, stdev
from math import log

def objective(data: list, entry: float, stop_loss: float, take_profit: float) -> float:
    mean_win = take_profit - entry - 1.5
    mean_loss = entry - stop_loss + 1.5
    num_trades = 0
    num_wins = 0

    entered = False
    for point in data:
        if entered == False:
             if point < entry:
                 entered = True
        else:
            if point > take_profit:
                num_wins += 1
                num_trades += 1
                entered = False

            elif point < stop_loss:
                num_trades += 1
                entered = False

    if num_trades > 0:
        win_rate = num_wins / num_trades
        profitability = (win_rate * mean_win - (1 - win_rate) * mean_loss)

        return profitability

    return 0

def monte_carlo(data: list):
    stdev_data = stdev(data)
    upper_band = 3 * stdev_data
    lower_band = -3 * stdev_data

    X1 = np.arange(lower_band, 0, 0.5)  #Entry
    X2 = np.arange(lower_band, 0, 0.5)  #Stop loss
    X3 = np.arange(0, upper_band, 0.5)  #Take profit

    X = []
    for x1 in X1:
        for x2 in X2:
            if x2 < x1:
                for x3 in X3:
                    if x3 > x1:
                        X.append([x1, x2, x3])

    y = [objective(data, x1, x2, x3) for x1, x2, x3 in X]
    ix = np.argmax(y)

    return X[ix][0], X[ix][1], X[ix][2]

def kalman_filter(y: list) -> list:
    outlier_thresh = 0.95
    observation_matrix = np.asarray([[1, 0]])

    x = []
    for n in range(1, len(y) + 1):
        x.append(n)

    y = np.transpose(np.asarray([y]))
    y = np.ma.array(y)

    dx = [np.mean(np.diff(x))] + list(np.diff(x))
    transition_matrices = np.asarray([[[1, each_dx],[0, 1]]
                                        for each_dx in dx])

    leave_1_out_cov = []

    for i in range(len(y)):
        y_masked = np.ma.array(copy.deepcopy(y))
        y_masked[i] = np.ma.masked

        kf1 = KalmanFilter(transition_matrices = transition_matrices,
                       observation_matrices = observation_matrix)

        kf1 = kf1.em(y_masked)

        leave_1_out_cov.append(kf1.observation_covariance[0,0])

    outliers = (leave_1_out_cov / np.mean(leave_1_out_cov)) < outlier_thresh

    for i in range(len(outliers)):
        if outliers[i]:
            y[i] = np.ma.masked


    kf1 = KalmanFilter(transition_matrices = transition_matrices,
                        observation_matrices = observation_matrix)

    kf1 = kf1.em(y)

    (smoothed_state_means, smoothed_state_covariances) = kf1.smooth(y)

    return [x[0] for x in smoothed_state_means]

if __name__ == '__main__':
    pass
