import streamlit as st 
import numpy as np
import altair as alt
import pandas as pd

col1, col2 = st.columns(2)
with col1:
    arrival_time = st.slider(label='Average Calls / Hour', value=12, min_value=1, max_value=60)
    days = st.number_input('Number of Days Simulated', value=100, min_value=10, max_value=500)
    random_seed = st.number_input('Random Seed', value=0, min_value=0)
with col2: 
    service_duration = st.slider('Mean Serivice Time (minutes / call)', value=6, min_value=1, max_value=20)
    agents = st.number_input(label='Number of Agents', value = 3, min_value=1, max_value=20)
    sla_threshold = st.slider('SLA Threshold (minutes)', value=2, min_value=1, max_value=10)

def get_arrival_times(ar_rate, total_hours, rng):
    mean_inter = 1 / ar_rate
    t = rng.exponential(mean_inter)
    arrivals = [t]
    while arrivals[-1] <= total_hours:
        t += rng.exponential(mean_inter)
        arrivals.append(t)
    if arrivals[-1] > total_hours:
        return arrivals[:-1]
    else:
        return arrivals

def get_mean_and_ci(metric):
    x = pd.Series(metric).astype(float)
    n = x.size
    m = x.mean()
    s = x.std(ddof=1) if n > 1 else 0.0
    half = 1.96 * s / np.sqrt(n) if n > 1 else np.nan
    return m, [m - half, m + half]

def aggregate_metrics(dfs, formatted=True):
    dfs = dfs.T

    dfs.columns = [
        'Arrival Rate (Calls / Hour)',
        'Service Rate per Agent (Calls / Hour)',
        'Agent Utilization',
        'Avg Wait Time (mins)',
        'Probability of Wait',
        'SLA %'
    ]

    arrival_rate, arrival_rate_ci = get_mean_and_ci(dfs['Arrival Rate (Calls / Hour)'])
    service_rate_per_agent, service_rate_per_agent_ci = get_mean_and_ci(dfs['Service Rate per Agent (Calls / Hour)'])
    agent_utilization, _ = get_mean_and_ci(dfs['Agent Utilization'])
    avg_wait_time, avg_wait_time_ci = get_mean_and_ci(dfs['Avg Wait Time (mins)'])
    probability_of_wait, probability_of_wait_ci = get_mean_and_ci(dfs['Probability of Wait'])
    sla_percentage, sla_percentage_ci = get_mean_and_ci(dfs['SLA %'])

    if formatted:
        final_metrics = pd.DataFrame({
        'Metrics':[
            'Arrival Rate (Calls / Hour)',
            'Service Rate per Agent (Calls / Hour)',
            'Agent Utilization',
            'Avg Wait Time (mins)',
            'Probability of Wait',
            'SLA %'
        ],
        'Values':[
            round(arrival_rate, 2),
            round(service_rate_per_agent, 2),
            round(agent_utilization, 2),
            round(avg_wait_time, 2),
            f'{probability_of_wait*100:.1f}%',
            f'{sla_percentage*100:.1f}%'
        ],
        'Confidence Intervals (95%)':[
            f'{arrival_rate_ci[0]:.2f} - {arrival_rate_ci[1]:.2f}', 
            f'{service_rate_per_agent_ci[0]:.2f} - {service_rate_per_agent_ci[1]:.2f}', 
            '--',
            f'{np.max([0, avg_wait_time_ci[0]]):.2f} - {avg_wait_time_ci[1]:.2f}',
            f'{np.max([0, probability_of_wait_ci[0]*100]):.1f}% - {probability_of_wait_ci[1]*100:.1f}%',
            f'{np.max([0, sla_percentage_ci[0]*100]):.1f}% - {np.min([100, sla_percentage_ci[1]*100]):.1f}%',
        ]
        })
    else:
        final_metrics = pd.DataFrame({
        'Metrics':[
            'Arrival Rate (Calls / Hour)',
            'Service Rate per Agent (Calls / Hour)',
            'Avg Wait Time (mins)',
            'Probability of Wait',
            'SLA %'
        ],
        'mean':[
            arrival_rate,
            service_rate_per_agent,
            avg_wait_time,
            probability_of_wait,
            sla_percentage
        ],
        'ci_low':[
            arrival_rate_ci[0], 
            service_rate_per_agent_ci[0], 
            np.max([0, avg_wait_time_ci[0]]),
            np.max([0, probability_of_wait_ci[0]]),
            np.max([0, sla_percentage_ci[0]]),
        ], 
        'ci_hi':[
            arrival_rate_ci[1], 
            service_rate_per_agent_ci[1], 
            avg_wait_time_ci[1],
            np.min([1, probability_of_wait_ci[1]]),
            np.min([1, sla_percentage_ci[1]]),
        ],
        'rho_mean':[
            agent_utilization, 
            agent_utilization,
            agent_utilization,
            agent_utilization, 
            agent_utilization
        ]
        })
    return final_metrics


def simulate_once(rng, seed, agents, arrival_rate, service_duration, sla_threshold=sla_threshold, days=days):
    arrival_times = get_arrival_times(arrival_rate, 8*days, rng)

    df = pd.DataFrame({
        'arrival_time':arrival_times
    })

    service_times = rng.exponential(service_duration / 60, size=len(arrival_times))
    df['service_time'] = service_times

    log_list = list(zip(arrival_times, service_times))

    agent_state = [0 for i in range(agents)]

    event_log = {
        'arrival_time': [],
        'service_time_length': [],
        'agent_id': [],
        'service_start': [],
        'service_end': [],
        'wait_time': []
    }

    def update_agent_state(agent_list, best_agent, service_end):
        return list(agent_list[:best_agent]) + [service_end] + list(agent_list[best_agent+1:])

    for event in log_list:
        arrival = event[0]
        service_time_length = event[1]
        best_agent = np.argmin(agent_state)
        agent_available_time = agent_state[best_agent]
        service_start = np.max([agent_available_time, arrival])
        service_end = service_start + service_time_length
        prev = agent_available_time
        assert service_end >= prev, "Agent timeline must be non-decreasing"
        assert service_end >= service_start, 'Cannot have service end before it begins'
        agent_state = update_agent_state(agent_state, best_agent, service_end)
        wait_time = np.max([0,service_start - arrival])
        assert wait_time >= 0, f"Negative wait: {wait_time}"
        event_log['arrival_time'].append(arrival)
        event_log['service_time_length'].append(service_time_length)
        event_log['agent_id'].append(best_agent)
        event_log['service_start'].append(service_start)
        event_log['service_end'].append(service_end)
        event_log['wait_time'].append(wait_time)
    

    test = pd.DataFrame(event_log)

    mean_interarrival = test['arrival_time'].diff().fillna(test.loc[0,'arrival_time']).mean()
    lambda_hat = 1 / mean_interarrival
    mean_service = test['service_time_length'].mean()
    mu_hat = 1 / mean_service


    rho = mean_service / (agents * mean_interarrival)
    avg_wait_time = (test['wait_time'] * 60).mean()
    prob_of_wait = (test['wait_time'] > 0).mean()
    sla_percentage = (test['wait_time'] * 60 <= sla_threshold).mean()

    metrics = pd.DataFrame({
        f'{seed}': [
            lambda_hat,
            mu_hat,
            rho,
            avg_wait_time,
            prob_of_wait,
            sla_percentage
        ]
    })
    return metrics

replications = 20
dfs = pd.DataFrame()
for num in range(random_seed, random_seed+replications):
    rng = np.random.default_rng(num)
    metrics = simulate_once(rng, num, agents, arrival_time, service_duration, days)
    dfs = pd.concat([dfs, metrics], axis=1)

final_metrics = aggregate_metrics(dfs, formatted=True)

st.dataframe(final_metrics)


st.divider()

st.markdown(
    '''
    ## Metrics by varying Agents
    '''
)


agents_df = pd.DataFrame()
for agents_count in range(2, 11):
    sweep = pd.DataFrame()
    for r in range(5):
        rng = np.random.default_rng(r)
        metrics = simulate_once(rng, r, agents_count, arrival_time, service_duration, 10)
        sweep = pd.concat([sweep, metrics], axis=1)

    agent_sweep_metrics = aggregate_metrics(sweep, formatted=False)
    agent_sweep_metrics['agent'] = agents_count
    agents_df = pd.concat([agents_df, agent_sweep_metrics])

agents_df = agents_df.reset_index()

# st.dataframe(agents_df)

metric_selection = st.selectbox(label='Select Metric to View', options=['SLA %', 'Probability of Wait', 'Avg Wait Time (mins)'])

chosen_metric_df = agents_df[agents_df['Metrics']==metric_selection]


base = alt.Chart(chosen_metric_df).encode(
    alt.X('agent:N', axis=alt.Axis(title='Agents')),
)

mean = base.mark_line(color='red').encode(
    alt.Y('mean:Q', axis=alt.Axis(format='%'))
)

# Create the first area chart with its own axis and color
ci = base.mark_area(opacity=0.3, color='#57A44C').encode(
    alt.Y('ci_low:Q', axis=alt.Axis(format='%')),
    alt.Y2('ci_hi:Q'),
    tooltip=[
        alt.Tooltip('mean:Q', title=f'Mean {metric_selection}', format='.1%'),
        alt.Tooltip('ci_low:Q', title='Lower CI', format='.1%'),
        alt.Tooltip('ci_hi:Q', title='Upper CI', format='.1%')
    ]
)

# Layer the two charts and resolve the scales to be independent
chart = alt.layer(ci, mean).resolve_scale(
    y='shared'
)

chart