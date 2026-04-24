import streamlit as st
from functions.load_matches import load_matches
from functions.load_table import load_table

st.title("Football Predictions App")

if st.button("Load data", type="primary"):
    with st.spinner("Loading matches and table..."):
        load_matches()
        load_table()
    st.success("Matches and table loaded!")

if 'matches' in st.session_state and 'table' in st.session_state:
    st.write(f"**Matches:** {len(st.session_state['matches'])} rows loaded")
    st.write(f"**Table:** {len(st.session_state['table'])} teams loaded")
