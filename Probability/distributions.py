import streamlit as st 
import scipy.stats as stats 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import altair as alt
from openai import OpenAI

st.header('Distribution Modeler')

st.subheader('Binomial Distribution - Modeling Basketball Free-Throw Likelihood?')
st.write('How likely is it that a given basketball player makes a certain number of Free-Throws')

ex = pd.DataFrame({
    'Player':['Steph Curry','Michael Jordan','Lebron James','Shaquille O\'neal'],
    'Historical FTA':[0.907, 0.835, 0.736, 0.527]
})

st.dataframe(ex)

ftp = st.slider('Player Free-Throw %', min_value=0.0, max_value=1.0, step=0.01, value=0.80)
shots = st.slider('Total number of shots', min_value=0, max_value=100, step=1, value=10)

pmf = [stats.distributions.binom.pmf(k=i, n=shots+1, p=ftp) for i in range(0, shots+1)]

cdf = [stats.distributions.binom.cdf(k=i, n=shots+1, p=ftp) for i in range(0, shots+1)]

df_cdf = pd.DataFrame({
    'x':range(0, shots+1),
    'y': cdf,
    'y1': 1 - np.array(cdf)
})

df_pmf = pd.DataFrame({
    'x':range(0, shots+1),
    'y': pmf
})

st.bar_chart(
    df_pmf, x='x', y='y',
    x_label=f'Number of Shots Made out of {shots}',
    y_label='Probability (%)'
)

options = [
    'What is the probability of making exactly k shots?',
    'What is the probability of making at least k shots?',
    'What is the probability of making at most k shots?'
]

question = st.selectbox(label='Choose your metric!', options=options)

def explain_binomial_distribution(shots, ftp, shots_made, shot_prob, prompt):
    
    client = OpenAI(
        # This is the default and can be omitted
        base_url='https://openrouter.ai/api/v1',
        api_key=st.secrets['openrouter']['llm_api_key'],
    )

    response = client.chat.completions.create(
        model="google/gemma-3n-e2b-it:free",

        messages=[
            {"role": "You are an analyst explaining statistical analysis to a stakeholder"},
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content

if question == options[0]:
    shots_made = st.slider("What's the probability that our player hits K shots?", min_value=0, max_value=shots, value=int(shots / 2))

    shot_prob = df_pmf[df_pmf['x']==shots_made].iloc[0,1]

    st.metric(value=str(round(shot_prob*100, 1))+'%', label=f'Probability to make {shots_made} shots')

    base = alt.Chart(df_pmf).add_params(
        alt.selection_interval()
    )

    line = base.mark_line().encode(x='x', y='y')

    # Vertical line
    vline = alt.Chart(pd.DataFrame({'x': [shots_made]})).mark_rule(color='red').encode(x='x')

    # Combine
    chart = line + vline

    button = st.button(label='Run randomized simulation to test theory')
    if button:
        rnd = np.random.binomial(n=shots, p=ftp, size=1)[0]  # Extract single value
        rnd2 = df_pmf[df_pmf['x']==rnd].iloc[0,1]
        
        # Create proper DataFrame for the point
        point_data = pd.DataFrame({'x': [rnd], 'y': [rnd2]})
        point = alt.Chart(point_data).mark_point(color='red', size=100).encode(
            x='x',
            y='y'
        )
        chart = line + vline + point

    # Display in Streamlit
    st.altair_chart(chart, use_container_width=True)

    prompt1 = f''' 
        a player takes {shots} shots with a {ftp:.2%} chance of success per shot.
        What does it mean if the probability of making exactly {shots_made} is {shot_prob:.2%}?
        In 100 words, explain why this makes sense.
    '''

    def fallback_exactly_k(k, n, p):
        return (
            f"This represents the probability that the player will make exactly {k} shots "
            f"out of {n} attempts, assuming a {p:.0%} chance of success for each shot. "
            f"It focuses on just one specific outcome among all the possible results."
        )


    try: 
        explanation = explain_binomial_distribution(shots, ftp, shots_made, shot_prob, prompt1)
        st.markdown("### 📘 AI-Powered Explanation")
        st.write(explanation)
    except Exception as e:
        st.markdown('### Explanation')
        st.write(fallback_exactly_k(shots_made, shots, ftp))

elif question == options[1]:
    shots_made = st.slider("", min_value=0, max_value=shots, value=int(shots / 2))

    shot_prob = df_cdf[df_cdf['x']==shots_made].iloc[0,2]

    st.metric(value=str(round(shot_prob*100, 1))+'%', label=f'Probability to make at least {shots_made} shots')

    base = alt.Chart(df_cdf).add_params(
        alt.selection_interval()
    )

    line = base.mark_line().encode(x='x', y='y1')

    vline = alt.Chart(pd.DataFrame({'x': [shots_made]})).mark_rule(color='red').encode(x='x')

    chart = line + vline

    button = st.button(label='Run randomized simulation to test theory')
    if button:
        rnd = np.random.binomial(n=shots, p=ftp, size=1)[0]  # Extract single value
        rnd2 = df_cdf[df_cdf['x']==rnd].iloc[0,2]
        
        # Create proper DataFrame for the point
        point_data = pd.DataFrame({'x': [rnd], 'y': [rnd2]})
        point = alt.Chart(point_data).mark_point(color='red', size=100).encode(
            x='x',
            y='y'
        )
        chart = line + vline + point

    # Display in Streamlit
    st.altair_chart(chart, use_container_width=True)

    prompt2 = f''' 
        a player takes {shots} shots with a {ftp:.2%} chance of success per shot.
        What does it mean if the probability of making at least {shots_made} is {shot_prob:.2%}?
        In 100 words, explain why this makes sense.
    '''

    def fallback_at_least_k(k, n, p):
        return (
            f"This is the probability that the player will make {k} or more successful shots "
            f"out of {n}, assuming each shot has a {p:.0%} chance of success. "
            f"It captures the likelihood of a strong or above-average performance."
        )

    try: 
        explanation = explain_binomial_distribution(shots, ftp, shots_made, shot_prob, prompt2)
        st.markdown("### 📘 AI-Powered Explanation")
        st.write(explanation)
    except Exception as e:
        st.markdown('### Explanation')
        st.write(fallback_at_least_k(shots_made, shots, ftp))

elif question == options[2]:
    shots_made = st.slider("", min_value=0, max_value=shots, value=int(shots / 2))

    shot_prob = df_cdf[df_cdf['x']==shots_made].iloc[0,1]

    st.metric(value=str(round((shot_prob)*100, 1))+'%', label=f'Probability to make at most {shots_made} shots')

    base = alt.Chart(df_cdf).add_params(
        alt.selection_interval()
    )

    area = alt.Chart(df_cdf[df_cdf['x']<=shots_made]).mark_area().encode(
        x='x', y='y'
    )

    line = base.mark_line().encode(x='x', y='y')

    vline = alt.Chart(pd.DataFrame({'x': [shots_made]})).mark_rule(color='red').encode(x='x')

    chart = line + vline + area

    button = st.button(label='Run randomized simulation to test theory')
    if button:
        rnd = np.random.binomial(n=shots, p=ftp, size=1)[0]  # Extract single value
        rnd2 = df_cdf[df_cdf['x']==rnd].iloc[0,1]
        
        # Create proper DataFrame for the point
        point_data = pd.DataFrame({'x': [rnd], 'y': [rnd2]})
        point = alt.Chart(point_data).mark_point(color='red', size=100).encode(
            x='x',
            y='y'
        )
        chart = line + vline + area + point

    # Display in Streamlit
    st.altair_chart(chart, use_container_width=True)

    prompt3 = f''' 
        a player takes {shots} shots with a {ftp:.2%} chance of success per shot.
        What does it mean if the probability of making at the most {shots_made} is {shot_prob:.2%}?
        In 100 words, explain why this makes sense.
    '''

    def fallback_at_most_k(k, n, p):
        return (
        f"This is the probability that the player will make no more than {k} successful shots "
        f"out of {n}, given a {p:.0%} success rate. "
        f"It includes all outcomes from 0 up to {k}, and often reflects the chance of a weak or unlucky performance."
    )

    try: 
        explanation = explain_binomial_distribution(shots, ftp, shots_made, shot_prob, prompt3)
        st.markdown("### 📘 AI-Powered Explanation")
        st.write(explanation)
    except Exception as e:
        st.markdown('### Explanation')
        st.write(fallback_at_most_k(shots_made, shots, ftp))