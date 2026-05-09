import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from functions.load_matches import load_matches
from functions.load_table import load_table

# ── Computation (runs once, cached in session state) ──────────────────────────

if 'matches' not in st.session_state or 'table' not in st.session_state:
    with st.spinner("Loading matches and table..."):
        load_matches()
        load_table()

df = st.session_state['matches'].copy()
df = df[df['Home Score'].astype(str).str.strip() != '']
df = df[df['Away Score'].astype(str).str.strip() != '']
df['Home Score'] = pd.to_numeric(df['Home Score'], errors='coerce')
df['Away Score'] = pd.to_numeric(df['Away Score'], errors='coerce')
df = df.dropna(subset=['Home Score', 'Away Score'])
df[['Home Score', 'Away Score']] = df[['Home Score', 'Away Score']].astype(int)

teams    = sorted(df['Home Team'].unique())
n        = len(teams)
team_idx = {t: i for i, t in enumerate(teams)}

home_idx   = df['Home Team'].map(team_idx).values
away_idx   = df['Away Team'].map(team_idx).values
home_goals = df['Home Score'].values
away_goals = df['Away Score'].values


def unpack(params):
    home_adv = params[0]
    attack   = np.concatenate([[0.0], params[1:n]])
    defense  = params[n:]
    return home_adv, attack, defense


def neg_log_likelihood(params):
    home_adv, attack, defense = unpack(params)
    lam_home = np.exp(home_adv + attack[home_idx] + defense[away_idx])
    lam_away = np.exp(           attack[away_idx] + defense[home_idx])
    ll = (poisson.logpmf(home_goals, lam_home) +
          poisson.logpmf(away_goals, lam_away))
    return -ll.sum()


if 'poisson_model' not in st.session_state:
    with st.spinner("Training Poisson model..."):
        n_params = 1 + (n - 1) + n
        x0       = np.zeros(n_params)
        result   = minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-12})
    if result.success or result.fun < neg_log_likelihood(np.zeros(len(x0))):
        home_adv, attack, defense = unpack(result.x)
        st.session_state['poisson_model'] = {
            'teams':    teams,
            'team_idx': team_idx,
            'home_adv': home_adv,
            'attack':   attack,
            'defense':  defense,
            'n_params': n_params,
            'result':   result,
        }
    else:
        st.error(f"Model optimisation did not converge: {result.message}")
        st.stop()

model    = st.session_state['poisson_model']
home_adv = model['home_adv']
attack   = np.array(model['attack'])
defense  = np.array(model['defense'])
team_idx = model['team_idx']

all_matches = st.session_state['matches'].copy()
played      = all_matches.copy()
played['Home Score'] = pd.to_numeric(played['Home Score'], errors='coerce')
played['Away Score'] = pd.to_numeric(played['Away Score'], errors='coerce')
played_mask = played['Home Score'].notna() & played['Away Score'].notna()

played_df   = played[played_mask].copy()
unplayed_df = all_matches[~played_mask].copy()

known_teams = set(team_idx.keys())
unplayed_df = unplayed_df[
    unplayed_df['Home Team'].isin(known_teams) &
    unplayed_df['Away Team'].isin(known_teams)
].reset_index(drop=True)

N_SIMS = 10000


def simulate_season(played_df, unplayed_df, home_adv, attack, defense, team_idx, n_sims):
    teams = list(team_idx.keys())
    h_idx = unplayed_df['Home Team'].map(team_idx).values
    a_idx = unplayed_df['Away Team'].map(team_idx).values
    lam_h = np.exp(home_adv + attack[h_idx] + defense[a_idx])
    lam_a = np.exp(           attack[a_idx] + defense[h_idx])

    rng = np.random.default_rng()
    hg  = rng.poisson(lam_h[:, None] * np.ones((1, n_sims)))
    ag  = rng.poisson(lam_a[:, None] * np.ones((1, n_sims)))

    base_pts = {t: 0 for t in teams}
    for _, row in played_df.iterrows():
        ht, at   = row['Home Team'], row['Away Team']
        hgs, ags = int(row['Home Score']), int(row['Away Score'])
        if ht in base_pts and at in base_pts:
            if hgs > ags:
                base_pts[ht] += 3
            elif ags > hgs:
                base_pts[at] += 3
            else:
                base_pts[ht] += 1
                base_pts[at] += 1

    base_arr = np.array([base_pts[t] for t in teams], dtype=float)
    sim_pts  = np.tile(base_arr[:, None], (1, n_sims))

    home_win = hg > ag
    away_win = ag > hg
    draw     = hg == ag

    for f in range(len(unplayed_df)):
        hi, ai = h_idx[f], a_idx[f]
        sim_pts[hi] += np.where(home_win[f], 3, np.where(draw[f], 1, 0))
        sim_pts[ai] += np.where(away_win[f], 3, np.where(draw[f], 1, 0))

    return pd.DataFrame(sim_pts, index=teams)


if 'simulation_results' not in st.session_state:
    if unplayed_df.empty:
        st.session_state['simulation_results'] = None
    else:
        with st.spinner(f"Running {N_SIMS} simulations..."):
            st.session_state['simulation_results'] = simulate_season(
                played_df, unplayed_df, home_adv, attack, defense, team_idx, N_SIMS
            )

# ── Display ───────────────────────────────────────────────────────────────────

st.title("Football Predictions App")

# ── Simulate Rest of Season (main section) ────────────────────────────────────

st.header("Rest of the season simulation")
st.write(f"**{len(played_df)}** matches played — **{len(unplayed_df)}** remaining")

with st.expander("Remaining fixtures"):
    st.dataframe(unplayed_df[['Round', 'Home Team', 'Away Team']], use_container_width=True)

pts_df = st.session_state['simulation_results']

if pts_df is None:
    st.info("No remaining matches to simulate — the season is complete.")
else:
    n_teams = pts_df.shape[0]
    ranks   = pts_df.rank(axis=0, ascending=False, method='min')

    winners    = pts_df.idxmax(axis=0)
    win_pct    = (winners.value_counts() / N_SIMS * 100).round(1)
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

    for place in [5, 6, 7]:
        pct  = ((ranks == place).sum(axis=1) / N_SIMS * 100).sort_values(ascending=False).round(1)
        p_df = pct.reset_index()
        p_df.columns = ['Team', f'{place}th Place %']
        p_df[f'{place}th Place %'] = p_df[f'{place}th Place %'].map(lambda x: f"{x:.1f}%")
        p_df.index += 1
        st.subheader(f"{place}th place probability")
        st.caption(f"Probability of finishing exactly {place}th.")
        st.dataframe(p_df, use_container_width=True)

    avg_df = pts_df.mean(axis=1).sort_values(ascending=False).reset_index()
    avg_df.columns = ['Team', 'Avg Points']
    avg_df['Avg Points'] = avg_df['Avg Points'].round(1)
    avg_df.index += 1

    st.subheader("Average final standings")
    st.caption("Mean points across all simulations.")
    st.dataframe(avg_df, use_container_width=True)

# ── Poisson model details (secondary) ────────────────────────────────────────

st.header("Poisson Model")
st.write(f"Trained on **{len(df)} played matches** across **{df['Home Team'].nunique()} teams**.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Home advantage (log scale)", f"{home_adv:.4f}")
    st.metric("Home advantage (goals multiplier)", f"{np.exp(home_adv):.3f}x")
with col2:
    st.metric("Matches used", len(df))
    st.metric("Teams", n)

params_df = pd.DataFrame({
    'Team':    model['teams'],
    'Attack':  attack,
    'Defense': defense,
}).sort_values('Attack', ascending=False).reset_index(drop=True)

with st.expander("Team attack & defense parameters"):
    st.caption("Attack: higher = scores more.  Defense: lower (more negative) = concedes less.")
    st.dataframe(params_df.style.format({'Attack': '{:.4f}', 'Defense': '{:.4f}'}),
                 use_container_width=True)

# ── Data summary (bottom) ─────────────────────────────────────────────────────

st.header("Data")
st.write(f"**{len(st.session_state['matches'])}** total matches loaded.")

if not st.session_state['table'].empty:
    with st.expander("Current league table"):
        st.dataframe(st.session_state['table'], use_container_width=True, hide_index=True)
