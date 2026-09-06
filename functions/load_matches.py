import streamlit as st
import pandas as pd
import requests


API_URL = "https://api.football-data.org/v4/competitions/PL/matches"
API_HEADERS = {"X-Auth-Token": "b006736168e34387975ae15e83b341a4"}


def load_matches():
    """Load the fixture list from the Football-Data API.

    Stores the result in st.session_state['matches'] using the same columns the
    model expects: Round, Home Team, Away Team, Home Score, Away Score.
    """
    try:
        response = requests.get(
            API_URL,
            headers=API_HEADERS,
            params={"season": 2026, "limit": 500},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        matches = data.get("matches", [])
        rows = []

        for match in matches:
            home_team = match.get("homeTeam", {}).get("shortName") or match.get("homeTeam", {}).get("name")
            away_team = match.get("awayTeam", {}).get("shortName") or match.get("awayTeam", {}).get("name")
            if not home_team or not away_team:
                continue

            score = match.get("score", {}) or {}
            full_time = score.get("fullTime") or {}
            status = match.get("status")

            if status == "FINISHED":
                home_score = full_time.get("home")
                away_score = full_time.get("away")
            else:
                home_score = ""
                away_score = ""

            round_text = match.get("matchday")
            if round_text is not None:
                round_text = f"Matchday {round_text}"

            rows.append({
                'Round': round_text or "",
                'Home Team': home_team,
                'Away Team': away_team,
                'Home Score': home_score if home_score is not None else "",
                'Away Score': away_score if away_score is not None else "",
            })

        if not rows:
            return "No matches returned by the API."

        st.session_state['matches'] = pd.DataFrame(rows)
        return None

    except Exception as e:
        return f"Failed to load matches from API: {e}"
