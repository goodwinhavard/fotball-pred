import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

st.header("Poisson Model Training")

# ── 1. Load & clean data ────────────────────────────────────────────────────

if 'matches' not in st.session_state:
    st.warning("No match data in session state. Go to the Matches page and load matches first.")
    st.stop()

df = st.session_state['matches'].copy()

# Keep only played matches (both scores present and numeric)
df = df[df['Home Score'].astype(str).str.strip() != '']
df = df[df['Away Score'].astype(str).str.strip() != '']
df['Home Score'] = pd.to_numeric(df['Home Score'], errors='coerce')
df['Away Score'] = pd.to_numeric(df['Away Score'], errors='coerce')
df = df.dropna(subset=['Home Score', 'Away Score'])
df[['Home Score', 'Away Score']] = df[['Home Score', 'Away Score']].astype(int)

st.subheader("Played matches used for training")
st.dataframe(df, use_container_width=True)
st.write(f"**{len(df)} matches** across **{df['Home Team'].nunique()} teams**")

# ── 2. Model definition ─────────────────────────────────────────────────────
#
# For each match (home team h, away team a):
#   Home goals ~ Poisson(λ_h),  log λ_h = home_adv + attack_h + defense_a
#   Away goals ~ Poisson(λ_a),  log λ_a =            attack_a + defense_h
#
# Parameters (2·N + 1):  home_adv, attack_0..N-1, defense_0..N-1
# Identifiability constraint: attack of the first team is fixed to 0
#   (i.e. it acts as the reference level)

teams = sorted(df['Home Team'].unique())
n = len(teams)
team_idx = {t: i for i, t in enumerate(teams)}

home_idx = df['Home Team'].map(team_idx).values
away_idx = df['Away Team'].map(team_idx).values
home_goals = df['Home Score'].values
away_goals = df['Away Score'].values


def unpack(params):
    home_adv = params[0]
    attack   = np.concatenate([[0.0], params[1:n]])   # attack[0] fixed to 0
    defense  = params[n:]
    return home_adv, attack, defense


def neg_log_likelihood(params):
    home_adv, attack, defense = unpack(params)
    lam_home = np.exp(home_adv + attack[home_idx] + defense[away_idx])
    lam_away = np.exp(           attack[away_idx] + defense[home_idx])
    ll = (poisson.logpmf(home_goals, lam_home) +
          poisson.logpmf(away_goals, lam_away))
    return -ll.sum()


# ── 3. Fit button ───────────────────────────────────────────────────────────

if st.button("Train Poisson model"):
    with st.spinner("Optimising..."):
        # n-1 free attack params (attack[0] fixed), n defense params, 1 home_adv
        n_params = 1 + (n - 1) + n
        x0 = np.zeros(n_params)

        result = minimize(
            neg_log_likelihood,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12}
        )

    if result.success or result.fun < neg_log_likelihood(np.zeros(len(x0))):
        home_adv, attack, defense = unpack(result.x)

        params_df = pd.DataFrame({
            'Team':    teams,
            'Attack':  attack,
            'Defense': defense,
        }).sort_values('Attack', ascending=False).reset_index(drop=True)

        model = {
            'teams':    teams,
            'team_idx': team_idx,
            'home_adv': home_adv,
            'attack':   attack,
            'defense':  defense,
            'n_params': n_params,
            'result':   result,
        }
        st.session_state['poisson_model'] = model

        st.success(f"Model trained. Log-likelihood: {-result.fun:.2f}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Home advantage (log scale)", f"{home_adv:.4f}")
            st.metric("Home advantage (goals multiplier)", f"{np.exp(home_adv):.3f}x")
        with col2:
            st.metric("Matches used", len(df))
            st.metric("Teams", n)

        st.subheader("Team attack & defense parameters")
        st.caption("Attack: higher = scores more.  Defense: lower (more negative) = concedes less.")
        st.dataframe(params_df.style.format({'Attack': '{:.4f}', 'Defense': '{:.4f}'}),
                     use_container_width=True)
    else:
        st.error(f"Optimisation did not converge: {result.message}")

elif 'poisson_model' in st.session_state:
    model = st.session_state['poisson_model']
    home_adv = model['home_adv']
    attack   = model['attack']
    defense  = model['defense']

    params_df = pd.DataFrame({
        'Team':    model['teams'],
        'Attack':  attack,
        'Defense': defense,
    }).sort_values('Attack', ascending=False).reset_index(drop=True)

    st.info("Showing previously trained model from session state.")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Home advantage (log scale)", f"{home_adv:.4f}")
        st.metric("Home advantage (goals multiplier)", f"{np.exp(home_adv):.3f}x")
    with col2:
        st.metric("Matches used", len(df))
        st.metric("Teams", n)

    st.subheader("Team attack & defense parameters")
    st.caption("Attack: higher = scores more.  Defense: lower (more negative) = concedes less.")
    st.dataframe(params_df.style.format({'Attack': '{:.4f}', 'Defense': '{:.4f}'}),
                 use_container_width=True)
