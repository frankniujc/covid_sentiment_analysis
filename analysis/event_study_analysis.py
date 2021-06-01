import numpy as np
import pandas as pd
from datetime import date as Date
from datetime import timedelta

from collections.abc import Iterable

from scipy.stats import ttest_ind, ttest_1samp, wilcoxon
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import  multipletests
import scipy

import matplotlib.pyplot as plt
import seaborn as sns
sns_palette = sns.color_palette()
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import matplotlib

from math import isnan, sqrt

plot_dir = '/home/frank/Desktop/paper_archive/smm4h2021/slides/plots'

def my_ztest(positive, n, null_hypothesis):
    # positive_ar, len(ar), 0.5
    p_0 = null_hypothesis
    p_hat = positive/n

    z = (p_hat - p_0) / sqrt((p_0 * (1-p_0)) / n)
    p_value = scipy.stats.norm.sf(abs(z))
    return z, p_value

def get_date_obj(date):
    if len(date) == 4:
        date_obj = Date(2020, int(date[:2]), int(date[2:]))
    else:
        date_obj = Date(int(date[:4]), int(date[4:6]), int(date[6:]))
    return date_obj

# Loading Data
def load_data(evi_path, ei_data_path, pt='can'):
    evi_dates = []
    evi = []
    with open(evi_path) as open_evi_file:
        for line in open_evi_file:
            if len(line.strip().split(',')) == 5:
                date, _, _, valence, _ = line.strip().split(',')
            elif len(line.strip().split(',')) == 3:
                date, valence, _ = line.strip().split(',')
            elif len(line.strip().split(',')) == 2:
                date, valence = line.strip().split(',')

            if len(date) == 4:
                date_obj = Date(2020, int(date[:2]), int(date[2:]))
            else:
                date_obj = Date(int(date[:4]), int(date[4:6]), int(date[6:]))

            evi_dates.append(date_obj)
            evi.append(float(valence))

    j, s, f, a = [], [], [], []
    v = []
    ei_dates = []
    date_tweet_count = []

    v_week_avg = []

    with open(ei_data_path) as open_file:

        for line in open_file:

            if len(line.strip().split(',')) == 2:
                date, valence = line.strip().split(',')
                ei_dates.append(get_date_obj(date))
                v.append(float(valence))
                date_tweet_count.append(None)

            elif len(line.strip().split(',')) == 3:
                date, valence, count = line.strip().split(',')
                
                if len(date) == 4:
                    date_obj = Date(2020, int(date[:2]), int(date[2:]))
                else:
                    date_obj = Date(int(date[:4]), int(date[4:6]), int(date[6:]))

                ei_dates.append(date_obj)
                v.append(float(valence))
                date_tweet_count.append(int(count))

            elif len(line.strip().split(',')) == 5:

                date, valence, _, _, count = line.strip().split(',')

                date_obj = get_date_obj(date)

                ei_dates.append(date_obj)
                v.append(float(valence))
                date_tweet_count.append(int(count))

            elif len(line.strip().split(',')) == 9:
                date, can_v, can_count, on_v, on_count, bc_v, bc_count, ab_v, ab_count = line.strip().split(',')
                ei_dates.append(get_date_obj(date))
                if pt == 'can':
                    v.append(float(can_v))
                    date_tweet_count.append(int(can_count))
                elif pt == 'on':
                    if on_v == 'None':
                        on_v = 'nan'
                    v.append(float(on_v))
                    date_tweet_count.append(int(on_count))
                elif pt == 'bc':
                    if bc_v == 'None':
                        bc_v = 'nan'
                    v.append(float(bc_v))
                    date_tweet_count.append(int(bc_count))
                elif pt == 'ab':
                    if ab_v == 'None':
                        ab_v = 'nan'
                    v.append(float(ab_v))
                    date_tweet_count.append(int(ab_count))
            else:

                date, joy, sadness, fear, anger, valence, count = line.strip().split(',')
                date_tweet_count.append(int(count))
                ei_dates.append(get_date_obj(date))
                j.append(float(joy))
                s.append(float(sadness))
                f.append(float(fear))
                a.append(float(anger))
                v.append(float(valence))

        for i, vv in enumerate(v):
            if i < 3 or i > len(v) - 3:
                v_week_avg.append(None)
            else:
                running_avg = sum(v[i-3:i+4]) / 7
                v_week_avg.append(running_avg)

    return (evi_dates, evi), (ei_dates, v, v_week_avg, date_tweet_count)


def market_model_fitting(evi, ei, estimation_period):

    # Get estimation period
    start, end = estimation_period

    evi_dates = evi[0]
    # ei_dates = ei[0]

    ei_dates = [x for x, y in zip(ei[0], ei[1]) if not isnan(y)]

    evi = list(zip(*evi))
    ei = list(zip(*ei))

    x, y = [], []

    for ei_ in ei:
        ei_date, v, v_week_avg, date_tweet_count = ei_
        if ei_date in evi_dates and ei_date >= start and ei_date <= end and not isnan(v):
            x.append(v)

    for evi_ in evi:
        evi_date, evi_v = evi_
        if evi_date in ei_dates and evi_date >= start and evi_date <= end:
            y.append(evi_v)

    ri = np.array(x)
    rm = np.array(y)

    cov_matrix = np.cov(ri, rm)
    cov = cov_matrix[0][1]
    std = cov_matrix[1][1]
    beta = cov / std

    alpha = (ri - beta * rm).sum() / len(rm)

    class Model:
        def __init__(self):
            self.alpha = alpha
            self.beta = beta

        def predict(self, x):
            return (beta * x.reshape((1, -1)))[0] + alpha

    model = Model()

    return model


def abnormal_return(evi, ei, model):

    # event_date, event_period = event
    predicted_ei = model.predict(np.array(evi[1]).reshape((-1, 1)))

    ei_dates = []
    ei_vals = []

    for ei_date, ei_val in zip(ei[0], ei[1]):
        if not isnan(ei_val):
            ei_dates.append(ei_date)
            ei_vals.append(ei_val)

    ei = (ei_dates, ei_vals)

    evi_overlapping_dates = [i for i, x in enumerate(evi[0]) if x in ei[0]]
    ei_overlapping_dates = [i for i, x in enumerate(ei[0]) if x in evi[0]]

    abnormal_returns = np.array(ei[1])[ei_overlapping_dates] - predicted_ei[evi_overlapping_dates]
    cumulated_abnormal_returns = np.array(list(map(lambda x: sum(abnormal_returns[:x]), range(len(abnormal_returns)))))

    return abnormal_returns, cumulated_abnormal_returns

def plot(evi, ei, events, model, abnormal_returns, measure_name):

    fig, axes = plt.subplots(3, 2, sharex='col', gridspec_kw={'width_ratios': [1, 10]})
    fig.set_size_inches(8, 8)

    # Valence Plots
    ax = axes[0,1]
    # Plot EVI
    line1 = sns.lineplot(*evi, ax=ax, label='Canadian valence index')

    # Plot the valence
    line2 = sns.lineplot(ei[0], ei[1], ax=ax, label=f'{measure_name} valence')
    line3 = sns.lineplot(ei[0], ei[2], ax=ax, label=f'{measure_name} valence-moving average')

    # Event time-line
    if not isinstance(events, Iterable):
        events = [events]
    for i, event_date in enumerate(events, 1):
        axes[0,1].axvline(event_date, color='black')
        ax.annotate(f'event {i}', xy=((event_date, ax.get_ylim()[1])), horizontalalignment='center')
        axes[1,1].axvline(event_date, color='black')
        axes[2,1].axvline(event_date, color='black')

    # Plot fitted data
    predicted_ei = model.predict(np.array(evi[1]).reshape((-1, 1)))
    sns.lineplot(evi[0], predicted_ei, ax=ax, label='model predicted valence')


    # CAR Plot
    ax = axes[1,1]
    ar, car = abnormal_returns


    ei_overlapping_dates = [x for x,y in zip(ei[0], ei[1]) if x in evi[0] and not isnan(y)]


    car -= car[ei_overlapping_dates.index(events[0])]

    ax.fill_between(ei_overlapping_dates, 0, car, label='CAR')
    sns.lineplot(ei_overlapping_dates, ar, color='purple', ax=ax, label='abnormal returns')
    ax.axhline(0, color='black')

    # Volume Plot
    ax = axes[2,1]
    sns.lineplot(ei[0], ei[3], ax=ax, label='tweet volume')
    ax.axhline(0, color='black')

    for i in range(3):
        axes[i,1].legend().remove()
        axes[i,0].text(0,0, '(i'+ 'i'*i +')')
        axes[i,0].axis('off')
        axes[i,0].set_ylim((-1,1))

    lgd = plt.figlegend(loc='lower center', ncol=2, borderaxespad=0)

    axes[0,1].set_ylabel('Twitter Sentiment')
    axes[1,1].set_ylabel('CAR')
    axes[2,1].set_ylabel('Tweet Volume')

    plt.subplots_adjust(bottom=0.15)

    months = mdates.MonthLocator()  # every month
    days = mdates.DayLocator()  # every month
    months_fmt = mdates.DateFormatter('%b')

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.xaxis.set_minor_locator(days)

    fig.savefig(f'{plot_dir}/{measure_name.replace(" ", "_")}.svg', bbox_extra_artists=(lgd,), bbox_inches='tight')
    return fig, axes



def plot_statistics(evi, ei, event_date, model, measure_name, event_name):

    t_p, w_p, z_p = [], [], []
    t_s, w_s, z_s = [], [], []

    after_event = 11

    for i in range(-10, after_event):
        t, w, r, z, _ = statistics(evi, ei, (event_date+timedelta(days=i), timedelta(days=4)), model)
        t_p.append(t[1])
        w_p.append(w[1])
        z_p.append(z[1])

        t_s.append(t[0])
        w_s.append(w[0])
        z_s.append(z[0])

    fig = plt.figure()
    ax = plt.gca()
    ax.grid()

    l = len(t_p)
    df = pd.DataFrame({
        'dates': list(range(-10,after_event))*3,
        'p': t_p + w_p + z_p,
        'p value': ['t-test']*l + ['Wilcoxon']*l + ['z-test']*l,
        })

    sns.barplot(data=df, x='dates', y='p', hue='p value', ax=ax)

    ax.axhline(0.05, color='black')
    ax.axhline(0.1, color='black')

    ax.set_xlabel('test window [from, to]')
    ax.set_ylabel('p-value')

    plt.xticks(range(21), [f'{x}\n{x+4}' for x in range(-10, 11)])
    ax.set_yscale('log')
    loc = matplotlib.ticker.FixedLocator([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1])
    ax.yaxis.set_major_locator(loc)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())


def plot_statistics_range(evi, ei, plot_range, model, measure_name, event_name):

    t_p, w_p, z_p = [], [], []
    t_s, w_s, z_s = [], [], []

    start, end = plot_range

    for day in pd.date_range(start,end):
        t, w, r, z, _ = statistics(evi, ei, (day, timedelta(days=4)), model)
        t_p.append(t[1])
        w_p.append(w[1])
        z_p.append(z[1])

        t_s.append(t[0])
        w_s.append(w[0])
        z_s.append(z[0])

    fig = plt.figure()
    ax = plt.gca()
    ax.grid()
    sns.lineplot(pd.date_range(start,end), t_p, ax=ax, label='t-test p value')
    sns.lineplot(pd.date_range(start,end), w_p, ax=ax, label='Wilcoxon test p value')
    sns.lineplot(pd.date_range(start,end), z_p, ax=ax, label='z-test p value')

    ax.axhline(0.05, color='black')
    ax.axhline(0.1, color='black')

    ax.set_xlabel('days')
    ax.set_ylabel('p-value')


def event_table(evi, ei, event_date, model, event_windows):
    for start, end in event_windows:
        t, w, r, z, carp = statistics(evi, ei, (event_date+timedelta(days=start), timedelta(days=end) - timedelta(days=start)), model)

        tt = t
        ww = w
        zz = z

    # Holm Bonferroni
    ts, ws, zs = [], [], []
    for i in range(start, end-3):

        t, w, _, z, _ = statistics(evi, ei, (event_date+timedelta(days=i), timedelta(days=4)), model)
        ts.append(t[1])
        ws.append(w[1])
        zs.append(z[1])

    ts.append(tt[1])
    ws.append(ww[1])
    zs.append(zz[1])

    _, tp, *_ = multipletests(ts, 0.05, 'holm')
    _, wp, *_ = multipletests(ws, 0.05, 'holm')
    _, zp, *_ = multipletests(zs, 0.05, 'holm')

    tp = tp[-1]
    wp = wp[-1]
    zp = zp[-1]

    print(f' {start},{end} & {carp*100:.2f} & {tt[0]:.2f} ({tp:.3f}) & {ww[0]} ({wp:.3f}) & {zz[0]:.2f} ({zp:.3f})')


def holm_bonferroni(evi, ei, event_date, model, event_windows):
    for start, end in event_windows:
        t, w, r, z, carp = statistics(evi, ei, (event_date+timedelta(days=start), timedelta(days=end) - timedelta(days=start)), model)

        print(f' {start},{end} & {carp*100:.2f} & {t[0]:.2f} {t[1]:.3f} & {w[0]:.2f} {w[1]:.3f} & {z[0]:.2f} {z[1]:.3f}')



def statistics(evi, ei, event, model, print_len=False):
    event_date, after_event = event
    start = event_date
    end = event_date + after_event

    start, end = sorted([start, end])

    predicted_ei = model.predict(np.array(evi[1]).reshape((-1, 1)))

    evi_overlapping_dates = [i for i, x in enumerate(evi[0]) if x in ei[0] and x >= start and x <= end]
    ei_overlapping_dates = [i for i, x in enumerate(ei[0]) if x in evi[0] and x >= start and x <= end]

    predicted = predicted_ei[evi_overlapping_dates]
    ei_ = np.array(ei[1])[ei_overlapping_dates]

    ar = ei_ - predicted

    positive_ar = len([x for x in ar if x > 0])

    ttest = ttest_1samp(ar, 0)
    wilcoxon_test = wilcoxon(ar)
    ratio = (positive_ar/len(ar))
    ztest = my_ztest(positive_ar, len(ar), 0.5)
    car_percentage = sum(ar) / sum(ei_)

    if print_len:
        print(start, end)

    return ttest, wilcoxon_test, ratio, ztest, car_percentage

def car_percent(evi, ei, event_info, abnormal_returns, measure_name):


    event_date, event_period = event_info
    start, end = event_period

    fig = plt.figure()
    ax = plt.gca()
    ax.grid()


    # CAR Plot
    ar, car = abnormal_returns
    ei_overlapping_dates = [x for x in ei[0] if x in evi[0]]
    ei_map = {k:v for k, v in zip(ei[0], ei[1])}

    cei = 0
    car = 0

    car_percentages = []

    for date, day_ar in zip(ei_overlapping_dates, ar):
        if event_date + timedelta(days=start) <= date <= event_date + timedelta(days=end):
            cei += ei_map[date]
            car += day_ar
            car_percentages.append(car / cei)

    sns.lineplot(range(start, end+1), car_percentages, ax=ax)

    ax.set_xlabel('days')
    ax.set_ylabel('CAR %')