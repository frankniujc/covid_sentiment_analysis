import pandas as pd
from datetime import timedelta, date as Date
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import gzip

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot

from event_study_analysis import *
sns_palette = sns.color_palette()

import matplotlib.dates as mdates

def parse_vox_data(vox_path, measure):
    vox_data = pd.read_csv(vox_path)

    periods = []
    measure_rates = []

    for _, row in vox_data.iterrows():
        start = row['start']
        start = Date(2020, start//100, start%100)
        end = row['end']
        end = Date(2020, end//100, end%100)
        periods.append((start,end))
        measure_rates.append(row[measure])
    return periods, measure_rates

def read_valence_data(periods, valence_path):
    data = {x:[] for x in periods}
    period_rank = {x:i for i,x in enumerate(periods)}

    with gzip.open(valence_path) as open_file:
        for line in open_file:
            _, _, _, date, _, v = line.decode().strip().split('\t')
            v = float(v)
            date = Date(2020, int(date[:2]), int(date[2:]))
            for period, lst in data.items():
                if period[0] <= date <= period[1]:
                    lst.append(v)

    vs = []
    for _ in range(27):
        vs.append(None)

    for period, lst in data.items():
        if not lst:
            mean = None
        else:
            mean = sum(lst) / len(lst)
        try:
            vs[period_rank[period]] = mean
        except IndexError:
            breakpoint()

    return vs

def car_in_period(periods, car, dates):
    data = {x:[] for x in periods[:23]}
    # period_rank = {x:i for i,x in enumerate(periods)}

    # for date, ar in zip(dates, ars):
    car_data = {date:c for date, c in zip(dates, car)}

    result = []

    for s, e in periods:
        if s in car_data and e in car_data:
            result.append(car_data[e])
        else:
            result.append(None)

    return result

def study(measure):

    # Pearson Correlation Plot

    fig, ax = plt.subplots()
    corr_plot = fig
    ax.grid()

    vox_path = 'data/vox_monitor.csv'
    periods, measure_rates = parse_vox_data(vox_path, measure)
    l1 = sns.lineplot(x=[x[0] for x in periods], y=measure_rates, ax=ax, color=sns_palette[0], label=f'VOX {measure} %')
    ax.legend().remove()


    estimation_period = Date(2020, 1, 1), Date(2021, 12, 21)
    evi, ei = load_data(
        'data/cvi.csv',
        f'data/{measure.replace(" ", "_")}.csv')

    market_model = market_model_fitting(evi, ei, estimation_period)
    ar, car = abnormal_return(evi, ei, market_model)
    ei_overlapping_dates = [x for x in ei[0] if x in evi[0]]

    ax.set_ylabel('Proportion of Compliance (%)')
    ax = ax.twinx()

    ax.set_ylabel('CAR')
    l2 = sns.lineplot(x=ei_overlapping_dates, y=car, ax=ax, color=sns_palette[1], label='CAR')


    months = mdates.MonthLocator()  # every month
    days = mdates.DayLocator()  # every month
    months_fmt = mdates.DateFormatter('%b')

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.xaxis.set_minor_locator(days)

    ax.legend().remove()

    plt.figlegend(loc='upper left')
    car_data = {date:c for date, c in zip(ei_overlapping_dates, car)}

    x, y, d = [], [], []
    for period, measure_rate in zip(periods, measure_rates):
        start, end = period
        for i in range((end - start).days):
            date = start + timedelta(days=i)
            if date in ei_overlapping_dates:
                x.append(car_data[date]*100)
                y.append(measure_rate)
                d.append(date)

    # Cross Correlation Plot
    fig, ax = plt.subplots()
    ax.grid()
    xcorr_result = ax.xcorr(x, y, usevlines=True, maxlags=80, normed=True, detrend=signal.detrend)
    lags, c, line, b = xcorr_result
    max_c = max(c)
    lag = lags[np.argmax(c)]



if __name__ == '__main__':
    study('wearing a mask')
    study('social distancing')
    plt.show()
