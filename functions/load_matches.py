import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup


def load_matches():
    try:
        url = "https://www.nifs.no/kamper.php?countryId=2&tournamentId=7&stageId=699613"
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        matches = soup.find_all('div', class_='kamp')
        data = []

        for match in matches:
            round_text = ""
            round_div = match.find_previous('div', string=lambda text: text and text.strip().startswith('Runde'))
            if round_div:
                round_text = round_div.get_text(strip=True)

            team_links = match.find_all('a', class_='nifs_laglink_nopad')
            if len(team_links) >= 2:
                home_team = team_links[0].get_text(strip=True)
                away_team = team_links[1].get_text(strip=True)
            else:
                continue

            res_div = match.find_next_sibling('div', class_='res')
            if res_div:
                score_link = res_div.find('a')
                if score_link:
                    score_text = score_link.get_text(strip=True)
                    score = score_text.split('(')[0].strip()
                    if '-' in score:
                        home_score, away_score = score.split('-')
                        home_score = home_score.strip()
                        away_score = away_score.strip()
                    else:
                        home_score = ''
                        away_score = ''
                else:
                    home_score = ''
                    away_score = ''
            else:
                home_score = ''
                away_score = ''

            data.append({
                'Round': round_text,
                'Home Team': home_team,
                'Away Team': away_team,
                'Home Score': home_score,
                'Away Score': away_score
            })

        if data:
            st.session_state['matches'] = pd.DataFrame(data)
        else:
            st.session_state['matches'] = pd.DataFrame()

    except Exception as e:
        st.error(f"Failed to scrape data: {e}")
        st.info("Make sure the URL is correct and try again.")
