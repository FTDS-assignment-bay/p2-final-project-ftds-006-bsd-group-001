# import libraries
import streamlit as st
import eda
import predict

# create sidebar
st.sidebar.markdown('# About')
st.sidebar.write('''Click here to see Exploratory Data Analysis and Investment Recommendations based on Risk Profile''')

# create navigation
navigation = st.sidebar.selectbox('Navigation',['Exploratory Data Analysis','Investment Recommendations'])

if navigation == 'Exploratory Data Analysis':
    eda.run()
else:
    predict.run()