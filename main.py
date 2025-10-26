import streamlit as st
import pandas as pd
import openai_helper.py
col1,col2=st.columns([3,2])
financial_data_df=pd.DataFrame({"Measure":["Company Name","Stock Symbol","Revenue","Net Income","EPS"],
                                "Value":["","","","",""]})
with col1:
    st.title("Financial Data Entity Extraction")
    news_article=st.text_area("Paste your financial news article data here",height=300)
    if st.button("Extract Entity Here"):
        financial_data_df=openai_helper.extract_financial_data(news_article)  
with col2:
    st.markdown("<br/>"*5,unsafe_allow_html=True)
    st.dataframe(financial_data_df,
    column_config={"Measure":st.column_config.Column(width=150),
    "Value":st.column_config.Column(width=150)},hide_index=True)
