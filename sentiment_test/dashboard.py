# dashboard.py
import streamlit as st
import psycopg2
import pandas as pd

conn = psycopg2.connect(
    dbname="newsdb",
    user="newsuser",
    password="news123",
    host="localhost",
    port=5432
)

st.title("📈 Real-Time Financial News Sentiment")

df = pd.read_sql("SELECT * FROM sentiment ORDER BY timestamp DESC LIMIT 200", conn)

st.dataframe(df)

sent_counts = df.groupby("sentiment").size()
st.bar_chart(sent_counts)