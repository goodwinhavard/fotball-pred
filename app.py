import numpy as np
import pandas as pd
import streamlit as st
from functions.load_matches import load_matches
from functions.load_table import load_table
from functions.train_poisson_model import train_poisson_model
from functions.simulate_season import split_matches, simulate_season

N_SIMS = 10000

# ── Computation (runs once, cached in session state) ──────────────────────────

if 'matches' not in st.session_state or 'table' not in st.session_state:
    with st.spinner("Loading matches and table..."):
        load_matches()
        load_table()

if 'poisson_model' not in st.session_state:
    with st.spinner("Training Poisson model..."):
        model, err = train_poisson_model(st.session_state['matches'])
    if err:
        st.error(f"Model training failed: {err}")
        st.stop()
    st.session_state['poisson_model'] = model

model       = st.session_state['poisson_model']
played_df, unplayed_df = split_matches(st.session_state['matches'], model['team_idx'])

if 'simulation_results' not in st.session_state:
    if unplayed_df.empty:
        st.session_state['simulation_results'] = None
    else:
        with st.spinner(f"Running {N_SIMS} simulations..."):
            pts_df, hg, ag = simulate_season(played_df, unplayed_df, model, N_SIMS)
            st.session_state['simulation_results'] = (pts_df, hg, ag)

# ── Display ───────────────────────────────────────────────────────────────────

st.title("Football Predictions App")

# ── Simulate Rest of Season (main section) ────────────────────────────────────

st.header("Rest of the season simulation")
st.write(f"**{len(played_df)}** matches played — **{len(unplayed_df)}** remaining")

with st.expander("Remaining fixtures"):
    st.dataframe(unplayed_df[['Round', 'Home Team', 'Away Team']], use_container_width=True)

sim_results = st.session_state['simulation_results']

if sim_results is None:
    pts_df = None
else:
    pts_df, hg, ag = sim_results

if pts_df is None:
    st.info("No remaining matches to simulate — the season is complete.")
else:
    encoded   = hg * 100 + ag
    modal_scores, modal_pcts = [], []
    for f in range(len(unplayed_df)):
        codes            = encoded[f]
        unique, counts   = np.unique(codes, return_counts=True)
        best             = np.argmax(counts)
        h, a             = divmod(int(unique[best]), 100)
        modal_scores.append(f"{h} - {a}")
        modal_pcts.append(round(counts[best] / N_SIMS * 100, 1))

    likely_df = unplayed_df[['Round', 'Home Team', 'Away Team']].copy()
    likely_df['Most Likely Score'] = modal_scores
    likely_df['Frequency %']       = [f"{p:.1f}%" for p in modal_pcts]
    likely_df = likely_df.reset_index(drop=True)
    likely_df.index += 1

    #with st.expander("Most likely result per fixture"):
    st.caption("The scoreline that occurred most often across all simulations.")
    st.dataframe(likely_df, use_container_width=True)


    n_teams  = pts_df.shape[0]
    # Add tiny noise to break point ties: ensures each simulation assigns
    # unique ranks so exactly 3 teams fall in the relegation zone per run.
    noise    = np.random.default_rng().random(pts_df.shape) * 1e-6
    pts_rank = pts_df + noise
    ranks    = pts_rank.rank(axis=0, ascending=False, method='first')

    win_pct     = (pts_rank.idxmax(axis=0).value_counts() / N_SIMS * 100).round(1)
    champion_df = win_pct.reset_index()
    champion_df.columns = ['Team', 'Win %']
    champion_df = champion_df.sort_values('Win %', ascending=False).reset_index(drop=True)
    champion_df.index += 1
    champion_df['Win %'] = champion_df['Win %'].map(lambda x: f"{x:.1f}%")

    st.subheader("Championship probability")
    st.caption(f"Based on {N_SIMS} simulations.")
    st.dataframe(champion_df, use_container_width=True)

    relg_pct = ((ranks > n_teams - 3).sum(axis=1) / N_SIMS * 100).sort_values(ascending=False).round(1)
    relg_df  = relg_pct.reset_index()
    relg_df.columns = ['Team', 'Relegation %']
    relg_df['Relegation %'] = relg_df['Relegation %'].map(lambda x: f"{x:.1f}%")
    relg_df.index += 1

    st.subheader("Relegation probability")
    st.caption("Probability of finishing in the bottom 3.")
    st.dataframe(relg_df, use_container_width=True)

    top5_pct = ((ranks <= 5).sum(axis=1) / N_SIMS * 100).sort_values(ascending=False).round(1)
    top5_df  = top5_pct.reset_index()
    top5_df.columns = ['Team', 'Top 5 %']
    top5_df['Top 5 %'] = top5_df['Top 5 %'].map(lambda x: f"{x:.1f}%")
    top5_df.index += 1
    st.subheader("Top 5 probability")
    st.caption("Probability of finishing in the top 5.")
    st.dataframe(top5_df, use_container_width=True)

    p67_pct = (((ranks == 6) | (ranks == 7)).sum(axis=1) / N_SIMS * 100).sort_values(ascending=False).round(1)
    p67_df  = p67_pct.reset_index()
    p67_df.columns = ['Team', '6th or 7th %']
    p67_df['6th or 7th %'] = p67_df['6th or 7th %'].map(lambda x: f"{x:.1f}%")
    p67_df.index += 1
    st.subheader("6th or 7th place probability")
    st.caption("Probability of finishing 6th or 7th.")
    st.dataframe(p67_df, use_container_width=True)

    p8_pct = ((ranks == 8).sum(axis=1) / N_SIMS * 100).sort_values(ascending=False).round(1)
    p8_df  = p8_pct.reset_index()
    p8_df.columns = ['Team', '8th Place %']
    p8_df['8th Place %'] = p8_df['8th Place %'].map(lambda x: f"{x:.1f}%")
    p8_df.index += 1
    st.subheader("8th place probability")
    st.caption("Probability of finishing exactly 8th.")
    st.dataframe(p8_df, use_container_width=True)

    avg_df = pts_df.mean(axis=1).sort_values(ascending=False).reset_index()
    avg_df.columns = ['Team', 'Avg Points']
    avg_df['Avg Points'] = avg_df['Avg Points'].round(1)
    avg_df.index += 1

    st.subheader("Average final standings")
    st.caption("Mean points across all simulations.")
    st.dataframe(avg_df, use_container_width=True)

# ── Poisson model details ─────────────────────────────────────────────────────

st.header("Poisson Model")
st.write(f"Trained on **{model['n_matches']} played matches** across **{len(model['teams'])} teams**.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Home advantage (log scale)", f"{model['home_adv']:.4f}")
    st.metric("Home advantage (goals multiplier)", f"{np.exp(model['home_adv']):.3f}x")
with col2:
    st.metric("Matches used", model['n_matches'])
    st.metric("Teams", len(model['teams']))

params_df = pd.DataFrame({
    'Team':    model['teams'],
    'Attack':  model['attack'],
    'Defense': model['defense'],
}).sort_values('Attack', ascending=False).reset_index(drop=True)

with st.expander("Team attack & defense parameters"):
    st.caption("Attack: higher = scores more.  Defense: lower (more negative) = concedes less.")
    st.dataframe(params_df.style.format({'Attack': '{:.4f}', 'Defense': '{:.4f}'}),
                 use_container_width=True)

# ── Data ──────────────────────────────────────────────────────────────────────

st.header("Data")
st.write(f"**{len(st.session_state['matches'])}** total matches loaded.")

if not st.session_state['table'].empty:
    with st.expander("Current league table"):
        st.dataframe(st.session_state['table'], use_container_width=True, hide_index=True)
