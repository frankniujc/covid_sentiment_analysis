from event_study_analysis import *

def social_distance_study(pt='can'):
    # Social distancing example

    # Social distancing measures released
    if pt == 'bc':
        event_date = Date(2020, 3, 17)
    elif pt == 'on':
        event_date = Date(2020, 3, 16)
    elif pt == 'ab':
        event_date = Date(2020, 3, 21)
    else:
        event_date = (Date(2020, 3, 11), Date(2020,3,21))
    estimation_period = Date(2020, 4, 1), Date(2020, 12, 31)

    evi, ei = load_data(
        'data/cvi.csv',
        'data/social_distancing.csv',
        pt)

    if pt != 'can':
        sd = pt.upper() + ' social distancing'
    else:
        sd = 'social distancing'

    market_model = market_model_fitting(evi, ei, estimation_period)
    print(market_model.alpha, market_model.beta)
    abnormal_returns = abnormal_return(evi, ei, market_model)

    plot(evi, ei, event_date, market_model, abnormal_returns, sd)

    if pt != 'can':
        plot_statistics(evi, ei, event_date, market_model, sd, '1')
    else:
        plot_statistics_range(evi, ei, event_date, market_model, sd, '1')

    if pt == 'on':
        event_table(evi, ei, event_date, market_model, [(2,7)])

    elif pt == 'bc':
        event_table(evi, ei, event_date, market_model, [(1,9)])

    elif pt == 'ab':
        event_table(evi, ei, event_date, market_model, [(3,9)])

def wearing_a_mask():

    estimation_period = Date(2020, 1, 1), Date(2020, 12, 31)

    event_dates = (
        Date(2020, 4, 6),  # Dr. Tam says non-medical masks can help stop the spread of COVID-19
        Date(2020, 5, 20),  # PHAC officially revise recommendation for people wearing mask
        )

    evi, ei = load_data(
        'data/cvi.csv',
        'data/wearing_a_mask.csv')

    market_model = market_model_fitting(evi, ei, estimation_period)
    print(market_model.alpha, market_model.beta)
    abnormal_returns = abnormal_return(evi, ei, market_model)

    plot(evi, ei, event_dates, market_model, abnormal_returns, 'wearing a mask')

    plot_statistics(evi, ei, event_dates[0], market_model, 'wearing a mask', 'april_6')
    plot_statistics(evi, ei, event_dates[1], market_model, 'wearing a mask', 'may_20')

    event_table(evi, ei, event_dates[1], market_model, [(1,9)])

if __name__ == '__main__':
    # Wearing a mask
    wearing_a_mask()

    # Social Distancing
    for i in ('on', 'bc', 'ab'):
        social_distance_study(i)
    social_distance_study()
    plt.show()