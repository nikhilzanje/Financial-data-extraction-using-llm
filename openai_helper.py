
import json
import os


import pandas as pd
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

try:
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
except Exception as e:
    print("Error initializing MistralClient. Make sure MISTRAL_API_KEY is set in your environment.")
    print(f"Details: {e}")
    exit()

def extract_financial_data(text):
    prompt=get_prompt_financial()+text
    response=client.chat.complete(
        model="mistral-tiny",
        messages=[{"role":"user","content":prompt}]
     
    )
    content=response.choices[0].message.content
    try:
        data=json.loads(content)
        return pd.DataFrame(data.items(),columns=["Measure","Value"])
    except(json.JSONDecodeError,IndexError):
        pass
    return pd.DataFrame({
        "Measure":["Company Name","Stock Symbol","Revenue","Netincome","EPS"],
        "Value":["","","",""]
    })

def get_prompt_financial():
    return """ Please retrieve company name,revenue,net income and earnings per share(a.k.a EPS) from the following
    news article.If you can't find the information from this article then return "Not found".Do not make things
    up.Then retrieve a stock symbol corresponding to that company.For this you can use your general knowledge
    (it doesn't have to be from this article ).Always return your response as valid JSON string.The format of
    that should be this,
    {
       "Company Name":"Walmart",
       "Stock Symbol":"WMT",
       "Revenue": 12.34 million",
       "Net Income":"34.78 million",
       "EPS":"2.1$"
}

News Article:-
=============
"""

if __name__=="__main__":
    text="""
     Tesla's earning news in text format:Tesla's earning this quarter blew all the estimates.They reported 4.5
     billion $ profit against a revenue of 30 billion $ Then earnings per share was 2.3 $
     """
    df=extract_financial_data(text)
    print(df.to_string())

