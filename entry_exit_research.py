from statistics import mean, stdev
from numpy import arange
from numpy import argmax

'''
p = n * (r * w - (1 - r) * l)
p = n * (r * w - l + rl)
p = nrw - n + nrl

p = profit
n = number of trades
r = win rate
w = average win
l = average loss

partial d of p with respect to n = rw-1+lr
partial d of p with respect to r = nw+ln

'''
def objective(data: list, entry: float, stop_loss: float, take_profit: float) -> float:
    mean_win = take_profit - entry - 2
    mean_loss = entry - stop_loss + 2
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

    win_rate = num_wins / num_trades
    profitability = num_trades * (win_rate * mean_win - (1 - win_rate) * mean_loss)

    return profitability

def find_optimal_levels(data: list):
    stdev_data = stdev(data)
    upper_band = 2 * stdev_data
    lower_band = -2 * stdev_data

    X1 = arange(lower_band, upper_band, 0.5)  #Entry
    X2 = arange(lower_band, upper_band, 0.5)  #Stop loss
    X3 = arange(lower_band, upper_band, 0.5)  #Take profit

    X = []
    for x1 in X1:
        for x2 in X2:
            if x2 < x1:
                for x3 in X3:
                    if x3 > x1:
                        X.append([x1, x2, x3])

    y = [objective(data, x1, x2, x3) for x1, x2, x3 in X]
    ix = argmax(y)

    return X[ix][0], X[ix][1], X[ix][2]
