import streamlit as st
from functions.load_matches import load_matches

st.header("Premier League Matches 2025-2026")

if st.button("Load matches"):
    with st.spinner("Scraping data from NIFS..."):
        load_matches()

if 'matches' in st.session_state:
    df = st.session_state['matches']
    if df.empty:
        st.info("No match data found on the page.")
    else:
        st.dataframe(df)
else:
    st.info("Click 'Load matches' to fetch match data.")
