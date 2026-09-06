import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup


def load_table():
    """Scrape the league table and store it in st.session_state['table'].

    Returns None on success (the table may legitimately be empty early in the
    season), or an error message string on failure. When an error is returned,
    st.session_state['table'] is left unset so the caller can abort before
    running the simulation.
    """
    try:
        url = "https://www.nifs.no/tabell.php?countryId=2&tournamentId=7&stageId=711178"
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table', class_='nifs_table_r')
        if not table:
            return "Could not find the league table on the page (the site layout may have changed)."

        data = []
        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < 10:
                continue  # skip header / empty rows

            position = cells[0].get_text(strip=True)
            if not position.isdigit():
                continue  # skip non-data rows

            # Team name is inside an anchor tag in the second cell
            team_cell = cells[1]
            team_link = team_cell.find('a')
            team = team_link.get_text(strip=True) if team_link else team_cell.get_text(strip=True)

            played    = cells[2].get_text(strip=True)
            won       = cells[3].get_text(strip=True)
            drawn     = cells[4].get_text(strip=True)
            lost      = cells[5].get_text(strip=True)
            goals_for = cells[6].get_text(strip=True)
            goals_aga = cells[8].get_text(strip=True)
            goal_diff = cells[9].get_text(strip=True)
            points    = cells[10].get_text(strip=True)

            data.append({
                'Pos':    int(position),
                'Team':   team,
                'P':      int(played)  if played.lstrip('-').isdigit()  else played,
                'W':      int(won)     if won.lstrip('-').isdigit()     else won,
                'D':      int(drawn)   if drawn.lstrip('-').isdigit()   else drawn,
                'L':      int(lost)    if lost.lstrip('-').isdigit()    else lost,
                'GF':     int(goals_for) if goals_for.lstrip('-').isdigit() else goals_for,
                'GA':     int(goals_aga) if goals_aga.lstrip('-').isdigit() else goals_aga,
                'GD':     goal_diff,
                'Pts':    int(points)  if points.lstrip('-').isdigit()  else points,
            })

        st.session_state['table'] = pd.DataFrame(data) if data else pd.DataFrame()
        return None

    except Exception as e:
        return f"Failed to scrape table: {e}"
