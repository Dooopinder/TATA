# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 23:45:26 2022

@author: Tejas
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import time
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objs  as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

st.set_page_config(page_title="Car Production Details",
                   page_icon=":chart_with_upwards_trend:",
                   layout="wide")
@st.cache
def get_data_from_excel():
    df=pd.read_excel(
       io='Book.xlsx',
       engine='openpyxl',
       sheet_name='Sheet1',
       usecols='A:R',
       nrows=24,
    )
    
    return df
df =get_data_from_excel()

st.markdown("""---""")

"Tata Production and Sales"


st.sidebar.header("Please Filter Here:")
date = st.sidebar.multiselect(
    "Select the Dates:",
    options=df["Date"].unique(),
    default=df["Date"].unique()
   
)

quarter = st.sidebar.multiselect(
    "Select the Quarter:",
    options=df["Quarters"].unique(),
    default=df["Quarters"].unique()
)

df_selection = df.query(
    "Date == @date &Quarters == @quarter"
)


    


st.dataframe(df_selection)

"MAIN STATS"

total_sales = int(df_selection["Total sales"].sum())
total_production = int(df_selection["Total Production"].sum())
total_domestic = int(df_selection["Domestic Total sales"].sum())
total_exports = int(df_selection["Total Exported sales"].sum())


left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Total Production:")
    st.subheader(f" {total_production:,}")
with middle_column:
    st.subheader("Domestic/Exports:")
    st.subheader(f"{total_domestic:,} || {total_exports:,}")
with right_column:
    st.subheader("Total Sales:")
    st.subheader(f" {total_sales:,}")
with right_column:
    st.subheader("Sales Error (Previous unsold car)")
    st.subheader(f"{total_sales-total_production :,}")

st.markdown("""---""")
" SALES BY PRODUCT"
sales_by_product_line = (
    df_selection.groupby(by=["Total Production"]).sum()[["Total sales"]].sort_values(by="Total sales")
)
fig_product_sales = px.bar(
    sales_by_product_line,
    x="Total sales",
    y=sales_by_product_line.index,
    orientation="v",
    title="<b>Sales by Product Line</b>",
    color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
    template="plotly",
)
fig_product_sales.update_layout(
    width=1000,
    height=1000,
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

st.plotly_chart(fig_product_sales)

"Production of India "
def get_data_from_excel():
    df=pd.read_excel(
       io='Book.xlsx',
       engine='openpyxl',
       sheet_name='Sheet2',
       usecols='A:K',
       nrows=7,
    )
    
    return df
df2 =get_data_from_excel()
st.markdown("""---""")
"ANNUAL"
st.sidebar.header("Please Filter Here:")
anually = st.sidebar.multiselect(
    "Select the Year:",
    options=df2["Anually"].unique(),
    default=df2["Anually"].unique()
)


df2_selection = df2.query(
    "Anually == @anually "
)
st.dataframe(df2_selection)

"National stats"

total_sales = int(df2_selection["Total sale by India"].sum())
total_production = int(df2_selection["India Production"].sum())
total_production_tata = int(df2_selection["Tata production"].sum())
total_domestic = int(df2_selection["India Domestic sales"].sum())
total_exports = int(df2_selection["India Exported sales"].sum())
total_domestic_tata = int(df2_selection["Tata domestic sales"].sum())
total_exports_tata = int(df2_selection["Tata exported sales"].sum())
total_sales_tata = int(df2_selection["Total sale by Tata"].sum())


left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("India Production:")
    st.subheader(f" {total_production:,}")
with left_column:
    st.subheader("India production excluding tata:")
    st.subheader(f" {total_production-total_production_tata:,}")
with middle_column:
    st.subheader("India Domestic sales/Exported sales:")
    st.subheader(f"{total_domestic:,} || {total_exports:,}")
with middle_column:
    st.subheader("India Domestic sales/Exported sales excluding tata:")
    st.subheader(f"{total_domestic-total_domestic_tata:,} || {total_exports-total_exports_tata:,}")
with right_column:
    st.subheader("India Exported sales:")
    st.subheader(f" {total_sales:,}")
with right_column:
    st.subheader("Total sale in India excluding tata")
    st.subheader(f"{total_sales-total_sales_tata:,}")

st.markdown("""---""")
" SALES BY PRODUCT"
sales_by_product_line = (
    df2_selection.groupby(by=["Production excluding Tata"]).sum()[["Sales excluding Tata"]].sort_values(by=["Sales excluding Tata"])
)
fig_product_sales = px.bar(
    sales_by_product_line,
    x="Sales excluding Tata",
    y=sales_by_product_line.index,
    orientation="v",
    title="<b>Sales by Product Line</b>",
    color_discrete_sequence=["#0083B8"] * len(sales_by_product_line),
    template="plotly",
)
fig_product_sales.update_layout(
    width=1000,
    height=1000,
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=(dict(showgrid=False))
)

st.plotly_chart(fig_product_sales)

st.markdown("""---""")
# Core Pkgs
import streamlit as st
import sklearn

import numpy as np 
import pickle as pkl


# Loading Models





def main():
	"""Very Simple Linear Regression App"""

	st.title(" Prediction of sales in 2021")

	html_templ = """
	<div style="background-color:cyan;padding:10px;">
    <h3 style="color:white">Very Simple Linear Regression Web App for Salary Determination</h3>
	</div>
	"""

	st.markdown(html_templ,unsafe_allow_html=True)
if __name__ == '__main__':
	main()
activity = ["Prediction","About"]
choice = st.sidebar.selectbox("Menu",activity)
if choice == 'Prediction':
    st.subheader("Prediction")
if st.button("Prediction"):
    work=np.array([[19172]])
    work2=np.array([[114784]])
    work3=np.array([[17127]])
    work4=np.array([[199634]])
    experience_reshaped = np.array([[199472.20002557]])
    experience_reshaped2 = np.array([[133730.09349061]])
    experience_reshaped3 = np.array([[180458.30231214]])
    experience_reshaped4 = np.array([[210151.35702865]])
    st.info("sales related to {} quarters of production: {}".format(work,(experience_reshaped.round(2))))
    st.info("sales related to {} quarters of production: {}".format(work2,(experience_reshaped2.round(2))))
    st.info("sales related to {} quarters of production: {}".format(work3,(experience_reshaped3.round(2))))
    st.info("sales related to {} quarters of production: {}".format(work4,(experience_reshaped4.round(2))))
            

# About CHOICE


