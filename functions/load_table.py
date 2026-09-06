import streamlit as st
import pandas as pd
import requests


def load_table():
    """Load the current Premier League table from the Football-Data API.

    Stores the result in st.session_state['table'].
    """
    try:
        uri = "https://api.football-data.org/v4/competitions/PL/standings"
        headers = {"X-Auth-Token": "b006736168e34387975ae15e83b341a4"}

        response = requests.get(uri, headers=headers, timeout=20)
        response.raise_for_status()

        data = response.json()
        standings = data.get("standings", [])
        if not standings:
            st.session_state['table'] = pd.DataFrame()
            return "No standings returned from the API."

        table = standings[0].get("table", [])
        df = pd.DataFrame([
            {
                'Pos': team['position'],
                'Team': team['team']['shortName'],
                'P': team['playedGames'],
                'W': team['won'],
                'D': team['draw'],
                'L': team['lost'],
                'GF': team['goalsFor'],
                'GA': team['goalsAgainst'],
                'GD': team['goalDifference'],
                'Pts': team['points'],
            }
            for team in table
        ])

        st.session_state['table'] = df
        return None

    except Exception as e:
        return f"Failed to load table from API: {e}"
