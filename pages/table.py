import streamlit as st
from functions.load_table import load_table

st.header("Premier League Table 2025-2026")

if st.button("Load table"):
    with st.spinner("Scraping data from NIFS..."):
        load_table()

if 'table' in st.session_state:
    df = st.session_state['table']
    if df.empty:
        st.info("No table data found on the page.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("Click 'Load table' to fetch the league table.")
