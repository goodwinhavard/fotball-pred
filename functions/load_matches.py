import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup


URL = "https://www.nifs.no/kamper.php?countryId=2&tournamentId=7&stageId=711178"


def _get_page_source():
    response = requests.get(
        URL,
        headers={
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/131.0.0.0 Safari/537.36'
            )
        },
        timeout=30,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    if soup.select('div.kamp'):
        return response.content

    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait

    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument(
        '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/131.0.0.0 Safari/537.36'
    )

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(URL)
        WebDriverWait(driver, 120).until(
            lambda current_driver: current_driver.find_elements(
                'css selector', 'div.kamp'
            )
        )
        return driver.page_source
    finally:
        driver.quit()


def load_matches():
    """Scrape the fixture list and store it in st.session_state['matches'].

    Returns None on success, or an error message string on failure. When an
    error is returned, st.session_state['matches'] is left unset so the caller
    can abort before running the simulation.
    """
    try:
        soup = BeautifulSoup(_get_page_source(), 'html.parser')

        matches = soup.find_all('div', class_='kamp')
        data = []

        print("Antall kamper:", len(matches))
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

        if not data:
            return "No matches found on the page (the site layout may have changed)."

        st.session_state['matches'] = pd.DataFrame(data)
        return None

    except Exception as e:
        return f"Failed to scrape matches: {e}"
