import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

st.header("Poisson Model Training")

with st.expander("Mathematical framework"):
    st.markdown("### Bivariate Poisson Regression for Football")
    st.markdown(
        "Goals scored in a football match are modelled as independent Poisson random variables. "
        "For a match between home team $h$ and away team $a$:"
    )
    st.latex(r"Y_h \sim \text{Poisson}(\lambda_h), \qquad Y_a \sim \text{Poisson}(\lambda_a)")
    st.markdown("The expected goals are linked to team-specific parameters via a **log-linear model**:")
    st.latex(r"""
\log \lambda_h = \mu + \alpha_h + \delta_a \\
\log \lambda_a = \alpha_a + \delta_h
""")
    st.markdown("""
| Symbol | Meaning |
|--------|---------|
| $\\mu$ | Home advantage (shared across all matches) |
| $\\alpha_i$ | Attack strength of team $i$ — higher means more goals scored |
| $\\delta_i$ | Defensive strength of team $i$ — lower means fewer goals conceded |
""")

    st.markdown("---")
    st.markdown("### Identifiability Constraint")
    st.markdown(
        "The model is over-parameterised without a constraint. "
        "We fix the attack parameter of the first team (alphabetically) to zero:"
    )
    st.latex(r"\alpha_0 = 0")
    st.markdown("This makes all other attack parameters relative to that reference team.")

    st.markdown("---")
    st.markdown("### Parameter Estimation")
    st.markdown(
        r"Parameters $\boldsymbol{\theta} = (\mu,\, \alpha_1, \dots, \alpha_{N-1},\, \delta_0, \dots, \delta_{N-1})$ "
        r"are estimated by **maximum likelihood**. The log-likelihood over $M$ matches is:"
    )
    st.latex(
        r"\ell(\boldsymbol{\theta}) = \sum_{m=1}^{M} \Bigl["
        r"\log p\!\left(y_h^{(m)} \mid \lambda_h^{(m)}\right)"
        r"+ \log p\!\left(y_a^{(m)} \mid \lambda_a^{(m)}\right)"
        r"\Bigr]"
    )
    st.markdown(
        r"where $\log p(k \mid \lambda) = k \log \lambda - \lambda - \log k!$ is the Poisson log-PMF. "
        "Optimisation is performed with **L-BFGS-B** (a quasi-Newton gradient method)."
    )

    st.markdown("---")
    st.markdown("### Prediction")
    st.markdown(
        r"For an unseen match $(h, a)$, the fitted $\hat{\lambda}_h$ and $\hat{\lambda}_a$ define full "
        "goal distributions. Match outcomes are simulated by drawing:"
    )
    st.latex(r"\tilde{Y}_h \sim \text{Poisson}(\hat{\lambda}_h), \qquad \tilde{Y}_a \sim \text{Poisson}(\hat{\lambda}_a)")
    st.markdown(
        "Running many such draws gives a probability distribution over scorelines, and from "
        "that: win/draw/loss probabilities, expected goals, and more."
    )

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
