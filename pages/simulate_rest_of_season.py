import streamlit as st
import pandas as pd
import numpy as np

st.header("Simulate Rest of Season")

# ── Guards ───────────────────────────────────────────────────────────────────

if 'matches' not in st.session_state:
    st.warning("No match data found. Load matches first.")
    st.stop()

if 'poisson_model' not in st.session_state:
    st.warning("No trained model found. Train the Poisson model first.")
    st.stop()

# ── Data prep ────────────────────────────────────────────────────────────────

all_matches = st.session_state['matches'].copy()
model       = st.session_state['poisson_model']

team_idx = model['team_idx']
home_adv = model['home_adv']
attack   = np.array(model['attack'])
defense  = np.array(model['defense'])

# Played matches (both scores numeric and non-empty)
played = all_matches.copy()
played['Home Score'] = pd.to_numeric(played['Home Score'], errors='coerce')
played['Away Score'] = pd.to_numeric(played['Away Score'], errors='coerce')
played_mask = played['Home Score'].notna() & played['Away Score'].notna()

played_df    = played[played_mask].copy()
unplayed_df  = all_matches[~played_mask].copy()

# Only keep unplayed games whose teams are in the trained model
known_teams  = set(team_idx.keys())
unplayed_df  = unplayed_df[
    unplayed_df['Home Team'].isin(known_teams) &
    unplayed_df['Away Team'].isin(known_teams)
].reset_index(drop=True)

st.write(f"**{len(played_df)}** matches played — **{len(unplayed_df)}** remaining to simulate")

if unplayed_df.empty:
    st.info("No remaining matches to simulate.")
    st.stop()

with st.expander("Remaining fixtures"):
    st.dataframe(unplayed_df[['Round', 'Home Team', 'Away Team']], use_container_width=True)

# ── Simulation ───────────────────────────────────────────────────────────────

N_SIMS = 10000

def simulate_season(played_df, unplayed_df, home_adv, attack, defense, team_idx, n_sims):
    """
    Returns a DataFrame (teams × n_sims) of final points per simulation.
    """
    teams = list(team_idx.keys())

    # Pre-compute λ for every unplayed fixture (vectorised)
    h_idx = unplayed_df['Home Team'].map(team_idx).values
    a_idx = unplayed_df['Away Team'].map(team_idx).values
    lam_h = np.exp(home_adv + attack[h_idx] + defense[a_idx])   # shape (F,)
    lam_a = np.exp(           attack[a_idx] + defense[h_idx])   # shape (F,)

    # Draw goals for all fixtures × all simulations at once
    rng         = np.random.default_rng()
    home_goals  = rng.poisson(lam_h[:, None] * np.ones((1, n_sims)))  # (F, S)
    away_goals  = rng.poisson(lam_a[:, None] * np.ones((1, n_sims)))  # (F, S)

    # Points from already-played matches (fixed, same in every sim)
    base_pts = {t: 0 for t in teams}
    for _, row in played_df.iterrows():
        ht, at = row['Home Team'], row['Away Team']
        hg, ag = int(row['Home Score']), int(row['Away Score'])
        if ht in base_pts and at in base_pts:
            if hg > ag:
                base_pts[ht] += 3
            elif ag > hg:
                base_pts[at] += 3
            else:
                base_pts[ht] += 1
                base_pts[at] += 1

    base_arr = np.array([base_pts[t] for t in teams], dtype=float)  # (N,)

    # Accumulate simulated points
    sim_pts = np.tile(base_arr[:, None], (1, n_sims))  # (N, S)

    home_win = home_goals > away_goals
    away_win = away_goals > home_goals
    draw     = home_goals == away_goals

    for f in range(len(unplayed_df)):
        hi, ai = h_idx[f], a_idx[f]
        sim_pts[hi] += np.where(home_win[f], 3, np.where(draw[f], 1, 0))
        sim_pts[ai] += np.where(away_win[f], 3, np.where(draw[f], 1, 0))

    return pd.DataFrame(sim_pts, index=teams)  # (N, S)


if st.button(f"Run {N_SIMS} simulations"):
    with st.spinner("Simulating…"):
        pts_df = simulate_season(
            played_df, unplayed_df, home_adv, attack, defense, team_idx, N_SIMS
        )
        st.session_state['simulation_results'] = pts_df

if 'simulation_results' not in st.session_state:
    st.info(f"Click the button above to run {N_SIMS} season simulations.")
    st.stop()

pts_df = st.session_state['simulation_results']

# ── Results ──────────────────────────────────────────────────────────────────

# Winner = team with most points in each simulation (ties: first alphabetically after sort)
winners = pts_df.idxmax(axis=0)   # series of length N_SIMS
win_counts = winners.value_counts()
win_pct    = (win_counts / N_SIMS * 100).round(1)

champion_df = (
    win_pct
    .reset_index()
    .rename(columns={'index': 'Team', 'count': 'Win %'})
    .sort_values('Win %', ascending=False)
    .reset_index(drop=True)
)
champion_df.index += 1
champion_df.columns = ['Team', 'Win %']
champion_df['Win %'] = champion_df['Win %'].map(lambda x: f"{x:.1f}%")

st.subheader("Championship probability")
st.caption(f"Based on {N_SIMS} simulations. Teams not shown won 0 simulations.")
st.dataframe(champion_df, use_container_width=True)

# ── Relegation probability ────────────────────────────────────────────────────

# Rank teams by points in each simulation (rank 1 = most points)
# Bottom 3 = rank > N - 3, i.e. the 3 lowest ranked teams
n_teams = pts_df.shape[0]
# rank(axis=0, ascending=False) gives rank 1 to the highest points per sim
ranks = pts_df.rank(axis=0, ascending=False, method='min')  # (N, S)
relegated_counts = (ranks > n_teams - 3).sum(axis=1)        # times each team was in bottom 3
relg_pct = (relegated_counts / N_SIMS * 100).sort_values(ascending=False).round(1)

relg_df = relg_pct.reset_index()
relg_df.columns = ['Team', 'Relegation %']
relg_df['Relegation %'] = relg_df['Relegation %'].map(lambda x: f"{x:.1f}%")
relg_df.index += 1

st.subheader("Relegation probability")
st.caption(f"Based on {N_SIMS} simulations. Probability of finishing in the bottom 3.")
st.dataframe(relg_df, use_container_width=True)

# ── 5th place probability ─────────────────────────────────────────────────────

fifth_counts = (ranks == 5).sum(axis=1).sort_values(ascending=False)
fifth_pct = (fifth_counts / N_SIMS * 100).round(1)

fifth_df = fifth_pct.reset_index()
fifth_df.columns = ['Team', '5th Place %']
fifth_df['5th Place %'] = fifth_df['5th Place %'].map(lambda x: f"{x:.1f}%")
fifth_df.index += 1

st.subheader("5th place probability")
st.caption(f"Based on {N_SIMS} simulations. Probability of finishing exactly 5th.")
st.dataframe(fifth_df, use_container_width=True)

# ── 6th place probability ─────────────────────────────────────────────────────

sixth_counts = (ranks == 6).sum(axis=1).sort_values(ascending=False)
sixth_pct = (sixth_counts / N_SIMS * 100).round(1)

sixth_df = sixth_pct.reset_index()
sixth_df.columns = ['Team', '6th Place %']
sixth_df['6th Place %'] = sixth_df['6th Place %'].map(lambda x: f"{x:.1f}%")
sixth_df.index += 1

st.subheader("6th place probability")
st.caption(f"Based on {N_SIMS} simulations. Probability of finishing exactly 6th.")
st.dataframe(sixth_df, use_container_width=True)

# ── 7th place probability ─────────────────────────────────────────────────────

seventh_counts = (ranks == 7).sum(axis=1).sort_values(ascending=False)
seventh_pct = (seventh_counts / N_SIMS * 100).round(1)

seventh_df = seventh_pct.reset_index()
seventh_df.columns = ['Team', '7th Place %']
seventh_df['7th Place %'] = seventh_df['7th Place %'].map(lambda x: f"{x:.1f}%")
seventh_df.index += 1

st.subheader("7th place probability")
st.caption(f"Based on {N_SIMS} simulations. Probability of finishing exactly 7th.")
st.dataframe(seventh_df, use_container_width=True)

# ── Average final table ───────────────────────────────────────────────────────

avg_pts = pts_df.mean(axis=1).sort_values(ascending=False)
avg_df  = avg_pts.reset_index()
avg_df.columns = ['Team', 'Avg Points']
avg_df['Avg Points'] = avg_df['Avg Points'].round(1)
avg_df.index += 1

st.subheader("Average final standings")
st.caption("Mean points across all simulations.")
st.dataframe(avg_df, use_container_width=True)
