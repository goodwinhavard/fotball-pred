import streamlit as st

st.header("Mathematical Framework")

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
