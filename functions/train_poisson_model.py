import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


def train_poisson_model(matches_df):
    """
    Fit a Poisson model on played matches.
    Returns (model_dict, error_message). On success error_message is None.
    """
    df = matches_df.copy()
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

    n_params = 1 + (n - 1) + n
    x0       = np.zeros(n_params)
    result   = minimize(neg_log_likelihood, x0, method='L-BFGS-B',
                        options={'maxiter': 1000, 'ftol': 1e-12})

    if not (result.success or result.fun < neg_log_likelihood(x0)):
        return None, result.message

    home_adv, attack, defense = unpack(result.x)
    model = {
        'teams':     teams,
        'team_idx':  team_idx,
        'home_adv':  home_adv,
        'attack':    attack,
        'defense':   defense,
        'n_params':  n_params,
        'n_matches': len(df),
        'result':    result,
    }
    return model, None
