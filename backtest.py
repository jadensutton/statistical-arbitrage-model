import math
import csv
import ftx_lib
import statsmodels

import numpy as np
import matplotlib.pyplot as plt
import tulipy as ti

from time import sleep
from statistics import mean, stdev
from models import monte_carlo, kalman_filter
from datetime import datetime, timedelta
from scipy.stats import linregress

pairs = {}
buy_and_hold_returns = []

client = ftx_lib.FtxClient(api_key='', api_secret='')

symbol_list = ['SRN-PERP', 'MTA-PERP', 'MNGO-PERP', 'FLM-PERP', 'BNT-PERP', 'ROOK-PERP', 'POLIS-PERP']

for ticker in symbol_list:
    start_date = int(datetime.strptime('2021-01-01 00:00:00.0000', "%Y-%m-%d %H:%M:%S.%f").timestamp())
    end_date = int(datetime.strptime('2021-12-31 00:00:00.0000', "%Y-%m-%d %H:%M:%S.%f").timestamp())

    klines = client.get_historical_data(ticker, 3600, 5000, start_date, end_date)

    candles = []
    for candle in klines:
        candles.append([candle['open'], candle['high'], candle['low'], candle['close']])

    if len(candles) > 0:
        buy_and_hold_return = (candles[-1][0] / candles[0][0] - 1) * 100
        buy_and_hold_returns.append(buy_and_hold_return)

        pairs[ticker] = {'candles': candles, 'trade_placed': False, 'size': 0, 'co_pair': None, 'entry': None, 'winning_trades': [], 'losing_trades': [], 'num_wins': 0, 'num_losses': 0, 'buy_and_hold_return': buy_and_hold_return}

num_candles = max([len(pairs[x]['candles']) for x in pairs])
removal_queue = []
for ticker in pairs:
    if len(pairs[ticker]['candles']) != num_candles:
        removal_queue.append(ticker)

for ticker in removal_queue:
    del pairs[ticker]

capital = [10000]

active_positions = []

buy_orders = {}
sell_orders = {}
exit_orders = {}
pair_trades = {}
for ticker in pairs:
    buy_orders[ticker] = []
    sell_orders[ticker] = []
    exit_orders[ticker] = []

num_days = num_candles / 24
years = num_days / 365
num_pairs = len(pairs)
active_trades = 0
max_active_trades = 10
default_position_sizing = 1 / max_active_trades
num_trades = 0
leverage = 1

total_wins = 0
total_losses = 0
win_returns = []
loss_returns = []
arb_returns = []

transaction_fee = 0.0007

def generate_entry_signal(i):
    pair_signals = []

    for ticker1 in pairs:
        for ticker2 in pairs:
            signals_generated = [x[0] for x in pair_signals] + [x[1] for x in pair_signals]
            if ticker1 != ticker2 and ticker1 not in signals_generated and ticker2 not in signals_generated and pairs[ticker1]['trade_placed'] == False and pairs[ticker2]['trade_placed'] == False:
                pair1_candles = pairs[ticker1]['candles'][i - 11:i + 1]
                pair2_candles = pairs[ticker2]['candles'][i - 11:i + 1]

                pair1_opens = [x[0] for x in pair1_candles]
                pair2_opens = [x[0] for x in pair2_candles]

                X = np.array(pair2_opens)
                Y = np.array(pair1_opens)

                coint_coef = linregress(X, Y)[0]

                spreads = []
                for n, open in enumerate(pair1_opens):
                    spreads.append(math.log(open) - coint_coef * math.log(pair2_opens[n]))

                kalman_spread = kalman_filter(spreads)[-1]
                gap = spreads[-1] - kalman_spread

                if gap < 0:
                    pair1 = ticker1
                    pair2 = ticker2
                else:
                    pair1 = ticker2
                    pair2 = ticker1

                pair1_candles = pairs[pair1]['candles'][i - 199:i + 1]
                pair2_candles = pairs[pair2]['candles'][i - 199:i + 1]

                pair1_opens = [x[0] for x in pair1_candles]
                pair2_opens = [x[0] for x in pair2_candles]

                z_scores = []
                for m in range(11, 200):
                    X = np.array(pair2_opens)
                    Y = np.array(pair1_opens)

                    coint_coef = linregress(X, Y)[0]

                    spreads = []
                    for n in range(m - 11, m + 1):
                        spreads.append(math.log(pair1_opens[n]) - coint_coef * math.log(pair2_opens[n]))

                    kalman_spread = kalman_filter(spreads)[-1]
                    stdev_spread = stdev(spreads)

                    gap = spreads[-1] - kalman_spread
                    z_score = gap / stdev_spread
                    z_scores.append(z_score)

                entry, stop_loss, take_profit = monte_carlo(z_scores)
                if z_scores[-1] < entry:
                    pair_signals.append([pair1, pair2, stop_loss, take_profit])

    return pair_signals

def generate_exit_signal(i, long_pair, short_pair, stop_loss, take_profit):
    long_pnl = pairs[long_pair]['candles'][i][0] / pairs[long_pair]['entry'] - 1
    short_pnl = 1 - pairs[short_pair]['candles'][i][0] / pairs[short_pair]['entry']
    net_position_pnl = long_pnl + short_pnl
    if net_position_pnl <= -0.5:
        return True

    long_pair_candles = pairs[long_pair]['candles'][i - 11:i + 1]
    short_pair_candles = pairs[short_pair]['candles'][i - 11:i + 1]

    long_pair_opens = [x[0] for x in long_pair_candles]
    short_pair_opens = [x[0] for x in short_pair_candles]

    X = np.array(short_pair_opens)
    Y = np.array(long_pair_opens)

    coint_coef = linregress(X, Y)[0]

    spreads = []
    for n, open in enumerate(long_pair_opens):
        spreads.append(math.log(open) - coint_coef * math.log(short_pair_opens[n]))

    kalman_spread = kalman_filter(spreads)[-1]
    stdev_spread = stdev(spreads)

    gap = spreads[-1] - kalman_spread
    z_score = gap / stdev_spread

    if z_score > take_profit or z_score < stop_loss:
        return True

    return False

def size_position(pair_name):
    if pair_name not in pair_trades:
        return 1
    else:
        sizing = (10 + mean(pair_trades[pair_name]['returns'])) / 10
        if sizing <= 0.1:
            sizing = 0.1

        return sizing


buy_and_hold_data = []
quantities = {}
for ticker in pairs:
    quantities[ticker] = 1 / pairs[ticker]['candles'][0][0]

first_trade = None
for i in range(num_candles):
    buy_and_hold_data.append(0)
    for ticker in pairs:
        buy_and_hold_data[-1] += pairs[ticker]['candles'][i][0] * quantities[ticker]

    capital.append(capital[-1])

    if i >= 199:
        exit_queue = []
        for long, short, stop_loss, take_profit in active_positions:
            exit_signal = generate_exit_signal(i, long, short, stop_loss, take_profit)
            if exit_signal == True:
                exit_queue.append([long, short])

                active_positions.remove([long, short, stop_loss, take_profit])

        for long, short in exit_queue:
            arb_return = 0
            sum_size = pairs[long]['size'] + pairs[short]['size']

            candles = pairs[long]['candles']
            curr_open = candles[i][0]

            exit = curr_open
            exit_price = exit * (1 - transaction_fee)

            exit_orders[long].append(i)

            entry_price = pairs[long]['entry']
            change = exit_price / entry_price
            capital[-1] = ((capital[-1] - pairs[long]['size']) + (change * pairs[long]['size']))

            trade_return = (change - 1) * 100
            arb_return += trade_return

            if exit_price > entry_price:
                print('WIN', long, 'Long',entry_price,'->',exit_price,trade_return)
                pairs[long]['winning_trades'].append(trade_return)
                win_returns.append(trade_return)
                total_wins += 1
                pairs[long]['num_wins'] += 1

            else:
                print('LOSS', long, 'Long',entry_price,'->',exit_price,trade_return)
                pairs[long]['losing_trades'].append(trade_return)
                loss_returns.append(trade_return)
                total_losses += 1
                pairs[long]['num_losses'] += 1

            pairs[long]['trade_placed'] = False
            active_trades -= 1

            candles = pairs[short]['candles']
            curr_open = candles[i][0]

            exit = curr_open

            exit_price = exit * (1 + transaction_fee)

            exit_orders[short].append(i)

            entry_price = pairs[short]['entry']
            change = 2 - exit_price / entry_price
            capital[-1] = ((capital[-1] - pairs[short]['size']) + (change * pairs[short]['size']))

            trade_return = (change - 1) * 100
            arb_return += trade_return

            if exit_price < entry_price:
                print('WIN', short, 'Short',entry_price,'->',exit_price,trade_return)
                pairs[short]['winning_trades'].append(trade_return)
                win_returns.append(trade_return)
                total_wins += 1
                pairs[short]['num_wins'] += 1

            else:
                print('LOSS', short, 'Short',entry_price,'->',exit_price,trade_return)
                pairs[short]['losing_trades'].append(trade_return)
                loss_returns.append(trade_return)
                total_losses += 1
                pairs[short]['num_losses'] += 1

            pairs[short]['trade_placed'] = False
            active_trades -= 1

            print('Final Arb Return {}%'.format(round(arb_return, 2)))
            arb_returns.append(arb_return)
            pair_name = '{pair1}/{pair2}'.format(pair1=long.replace('-PERP', ''), pair2=short.replace('-PERP', ''))
            if pair_name not in pair_trades:
                pair_trades[pair_name] = {'returns': [arb_return], 'num_trades': 1}
            else:
                pair_trades[pair_name]['returns'].append(arb_return)
                pair_trades[pair_name]['num_trades'] += 1

        pair_signals = generate_entry_signal(i)

        for long, short, stop_loss, take_profit in pair_signals:
            if active_trades < max_active_trades:
                pair_name = '{pair1}/{pair2}'.format(pair1=long.replace('-PERP', ''), pair2=short.replace('-PERP', ''))
                position_sizing = size_position(pair_name)

                candles = pairs[long]['candles']
                curr_open = candles[i][0]

                size = position_sizing * leverage * default_position_sizing * capital[-1]
                entry_price = curr_open * (1 + transaction_fee)
                pairs[long]['entry'] = entry_price
                pairs[long]['size'] = size
                pairs[long]['qty'] = size / curr_open

                buy_orders[long].append(i)

                num_trades += 1
                active_trades += 1
                pairs[long]['trade_placed'] = True
                pairs[long]['type_of_trade'] = 'Long'
                pairs[long]['co_pair'] = short

                print('BUY ORDER', long)

                if first_trade == None:
                    first_trade = i

                candles = pairs[short]['candles']
                curr_open = candles[i][0]

                size = position_sizing * leverage * default_position_sizing * capital[-1]
                entry_price = curr_open * (1 - transaction_fee)
                pairs[short]['entry'] = entry_price
                pairs[short]['size'] = size
                pairs[short]['qty'] = size / curr_open

                sell_orders[short].append(i)

                num_trades += 1
                active_trades += 1
                pairs[short]['trade_placed'] = True
                pairs[short]['type_of_trade'] = 'Short'
                pairs[short]['co_pair'] = long

                print('SELL ORDER', short)

                active_positions.append([long, short, stop_loss, take_profit])

del capital[0]
capital = capital[first_trade:]
buy_and_hold_data = buy_and_hold_data[first_trade:]

returns = []
weekly_returns = []
weekly_buy_and_hold_returns = []
drawdown_periods = [0]
in_drawdown = False
local_maximum = 0
for n in range(len(capital)):
    if (n + 1) % 168 == 0:
        weekly_returns.append(capital[n] / capital[n - 167] - 1)
        weekly_buy_and_hold_returns.append(buy_and_hold_data[n] / buy_and_hold_data[n - 167] - 1)

    if n > 0:
        returns.append(capital[n] / capital[n - 1] - 1)
        curr_value = capital[n]
        prev_value = capital[n - 1]
        if in_drawdown == True:
            if curr_value > local_maximum:
                in_drawdown = False
            else:
                drawdown_periods[-1] += 1

        elif curr_value < prev_value:
            in_drawdown = True
            local_maximum = prev_value
            drawdown_periods.append(1)

max_drawdown = 100 * max(drawdown_periods) / len(capital)
sharpe = math.sqrt(52) * mean(weekly_returns) / stdev(weekly_returns)
underlying_sharpe = math.sqrt(52) * mean(weekly_buy_and_hold_returns) / stdev(weekly_buy_and_hold_returns)
sortino = math.sqrt(52) * mean(weekly_returns) / stdev([x for x in weekly_returns if x < 0])
underlying_sortino = math.sqrt(52) * mean(weekly_buy_and_hold_returns) / stdev([x for x in weekly_buy_and_hold_returns if x < 0])
period_return = (capital[-1] / capital[0] - 1) * 100
annualized_return = (((capital[-1] / capital[0]) ** (1 / years)) - 1) * 100

print('\nBacktest Data')

print('\nGeneral Statistics')
print('\tPeriod Return: {}%'.format(round(period_return, 2)))
print('\tBuy and Hold Return: {}%'.format(round(buy_and_hold_return, 2)))
print('\tSharpe Ratio: {}'.format(round(sharpe, 2)))
print('\tUnderlying Sharpe Ratio: {}'.format(round(underlying_sharpe, 2)))
print('\tSortino Ratio: {}'.format(round(sortino, 2)))
print('\tUnderlying Sortino Ratio: {}'.format(round(underlying_sortino, 2)))
print('\tMax Drawdown: {}%'.format(round(max_drawdown, 2)))
print('\tWin Rate: {}%'.format(round(100 * total_wins / (total_wins + total_losses), 2)))
print('\tW/L Ratio: {}'.format(round(abs(mean(win_returns) / mean(loss_returns)), 2)))
print('\tMean Win: {}%'.format(round(mean(win_returns), 2)))
print('\tMean Loss: {}%'.format(round(mean(loss_returns), 2)))
print('\tTrades: {}'.format(num_trades))
print('\tMean Arb Return: {}%'.format(round(mean(arb_returns), 2)))
print('\tAverage Annualized Return: {}%'.format(round(annualized_return, 2)))
print('\nReturns of a ${investment} Investment in Strat Over {years} Years'.format(investment=capital[0], years=round(years, 2)))
print('\tStrat: ${}'.format(round(capital[-1], 2)))

print('\nPair Results')
for pair in pair_trades:
    print(pair)
    print('\tTrades: {}'.format(pair_trades[pair]['num_trades']))
    print('\tMean Return: {}'.format(round(mean(pair_trades[pair]['returns']), 2)))

plt.figure('Returns of Algorithm')
plt.title('Performance of a ${} Investment in the Strat'.format(capital[0]))
plt.xlabel('Trading Periods')
plt.ylabel('Performance (%)')
plt.plot(100 * (np.array(capital) / capital[0] - 1), label='Capital')
plt.show()
