# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:33:31 2022

@author: Tejas
"""

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Car Production Details",
                   page_icon=":chart_with_upwards_trend:",
                   layout="wide")
@st.cache
def get_data_from_excel():
    df=pd.read_excel(
       io='Book.xlsx',
       engine='openpyxl',
       sheet_name='Sheet1',
       usecols='A:Q',
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


df_selection = df.query(
    "Date == @date "
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
import joblib,os
import numpy as np 
import tensorflow as tf
import keras as kr
from keras.models import Sequential
from keras.layers import Dense, Activation

import joblib

# Loading Models
def load_prediction_model(Prediction_model):
	loaded_model = joblib.load(open(os.path.join(Prediction_model),"rb"))
	return loaded_model



def main():
	"""Very Simple Multi Linear Regression App"""

	st.title(" Prediction of sales")

	html_templ = """
	<div style="background-color:cyan;padding:10px;">
    <h3 style="color:white">Very Simple Linear Regression Web App for Salary Determination</h3>
	</div>
	"""

	st.markdown(html_templ,unsafe_allow_html=True)

	activity = ["Prediction","About"]
	choice = st.sidebar.selectbox("Menu",activity)

# Salary Determination CHOICE
	if choice == 'Prediction':

		st.subheader("Prediction")

		experience = st.slider("Years ",2016,2022)

		#st.write(type(experience))

		if st.button("Determination"):

			regressor = load_prediction_model(r"Prediciton_model.pkl")
			experience_reshaped = np.array(experience).reshape(-1,1)

			#st.write(type(experience_reshaped))
			#st.write(experience_reshaped.shape)

			predicted_sales = regressor.predict(experience_reshaped)

			st.info("sales related to {} years of production: {}".format(experience,(predicted_sales[0][0].round(2))))

# About CHOICE
	if choice == 'About':
		st.subheader("About")
		st.markdown("""
			## Very Multiple Linear Regression App
			
			##### By
			+ *Tejas Verma
			""")


if __name__ == '__main__':
	main()