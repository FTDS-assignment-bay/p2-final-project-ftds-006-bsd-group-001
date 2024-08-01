# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import json
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model



# Save Pipeline
with open('features.txt', 'r') as file_1:
    final_features = json.load(file_1)

with open('class_names.txt', 'r') as file_5:
    class_names = json.load(file_5)

with open('pca.pkl', 'rb') as file_2:
    pca = pickle.load(file_2)

with open('pipeline.pkl', 'rb') as file_3:
    pipeline = pickle.load(file_3)

with open('cluster.pkl', 'rb') as file_4:
    kmeans = pickle.load(file_4)

model = load_model('model_seq.keras')

def run():
    # Title
    st.title('Investment Instrument Suggestion')
    st.markdown('---')
    
    # Form
    with st.form(key='questions'):

        col1, col2 = st.columns(2)
        
        # with col1:
        
        with col1:
            age = st.number_input("Age", min_value=16, max_value=70, placeholder="Type a number...")
        
        with col2:
            allowance = st.number_input("Allowance (in IDR)", min_value=0, max_value=1000000000, placeholder="Type a number...")
        
        gender = st.radio("Gender", ["Male", "Female"], index=None)

        container_1 = st.container(border=True)

        container_1.write("## Financial Literacy")
        options = ["Disagree", "Agree", "Neutral", "Strongly Agree", "Very Strongly Agree"]

        q1 = container_1.radio(f' 1 . I have a better understanding of how to invest my money \n \n *Saya memiliki cukup pemahaman tentang berinvestasi*', options, key=f'question_1')
        q2 = container_1.radio(f' 2 . I have a better understanding of how to manage my credit use \n \n *Saya memiliki pemahaman yang cukup dalam menggunakan kartu kredit*', options, key=f'question_2')
        q3 = container_1.radio(f' 3 . I have the ability to maintain financial records for my income and expenditure \n \n *Saya melakukan pencatatan keuangan saya terkait pemasukan dan pengeluaran*', options, key=f'question_3')
        q4 = container_1.radio(f'  4 . I can manage my money easily \n \n *Saya dapat mengatur keuangan dengan mudah*', options, key=f'question_4')
        q5 = container_1.radio(f'  5 . I have a better understanding of financial instruments (e.g. Bonds, stock, T-bill, time value of money, future contract, option, etc.) \n \n *Saya memiliki pemahaman yang cukup baik pada instrumen keuangan*', options, key=f'question_5')
        q6 = container_1.radio(f'  6 . I have the ability to prepare my own budget weekly and monthly \n \n *Saya memiliki kemampuan untuk membuat anggaran saya sendiri setiap minggu dan setiap bulan*', options, key=f'question_6')
        
        container_2 = st.container(border=True)

        container_2.write("## Self-Control")

        q7 = container_2.radio(f' 7 . I have a better understanding of how to invest my money \n \n *Saya memiliki cukup pemahaman tentang berinvestasi*', options, key=f'question_7')
        q8 = container_2.radio(f' 8 . I have a better understanding of how to manage my credit use \n \n *Saya memiliki pemahaman yang cukup dalam menggunakan kartu kredit*', options, key=f'question_8')
        q9 = container_2.radio(f' 9 . I have the ability to maintain financial records for my income and expenditure \n \n *Saya melakukan pencatatan keuangan saya terkait pemasukan dan pengeluaran*', options, key=f'question_9')
        q10 = container_2.radio(f'  10 . I can manage my money easily \n \n *Saya dapat mengatur keuangan dengan mudah*', options, key=f'question_10')
        q11 = container_2.radio(f'  11 . I have a better understanding of financial instruments (e.g. Bonds, stock, T-bill, time value of money, future contract, option, etc.) \n \n *Saya memiliki pemahaman yang cukup baik pada instrumen keuangan*', options, key=f'question_11')
        q12 = container_2.radio(f'  12 . I have the ability to prepare my own budget weekly and monthly \n \n *Saya memiliki kemampuan untuk membuat anggaran saya sendiri setiap minggu dan setiap bulan*', options, key=f'question_12')
        q13 = container_2.radio(f'  13 . I have the ability to prepare my own budget weekly and monthly \n \n *Saya memiliki kemampuan untuk membuat anggaran saya sendiri setiap minggu dan setiap bulan*', options, key=f'question_13')
        q14 = container_2.radio(f'  14 . I have the ability to prepare my own budget weekly and monthly \n \n *Saya memiliki kemampuan untuk membuat anggaran saya sendiri setiap minggu dan setiap bulan*', options, key=f'question_14')
        q15 = container_2.radio(f'  15 . I have the ability to prepare my own budget weekly and monthly \n \n *Saya memiliki kemampuan untuk membuat anggaran saya sendiri setiap minggu dan setiap bulan*', options, key=f'question_15')

        container_3 = st.container(border=True)

        container_3.write("## Peer-Influence")

        q16 = container_3.radio(f' 16 . I have a better understanding of how to invest my money \n \n *Saya memiliki cukup pemahaman tentang berinvestasi*', options, key=f'question_16')
        q17 = container_3.radio(f' 17 . I have a better understanding of how to manage my credit use \n \n *Saya memiliki pemahaman yang cukup dalam menggunakan kartu kredit*', options, key=f'question_17')
        q18 = container_3.radio(f' 18 . I have the ability to maintain financial records for my income and expenditure \n \n *Saya melakukan pencatatan keuangan saya terkait pemasukan dan pengeluaran*', options, key=f'question_18')
        q19 = container_3.radio(f'  19 . I can manage my money easily \n \n *Saya dapat mengatur keuangan dengan mudah*', options, key=f'question_19')
        q20 = container_3.radio(f'  20 . I have a better understanding of financial instruments (e.g. Bonds, stock, T-bill, time value of money, future contract, option, etc.) \n \n *Saya memiliki pemahaman yang cukup baik pada instrumen keuangan*', options, key=f'question_20')
        q21 = container_3.radio(f'  21 . I have the ability to prepare my own budget weekly and monthly \n \n *Saya memiliki kemampuan untuk membuat anggaran saya sendiri setiap minggu dan setiap bulan*', options, key=f'question_21')

        container_4 = st.container(border=True)

        container_4.write("## Investment Behavior")

        q22 = container_4.radio(f' 22 . I have a better understanding of how to invest my money \n \n *Saya memiliki cukup pemahaman tentang berinvestasi*', options, key=f'question_22')
        q23 = container_4.radio(f' 23 . I have a better understanding of how to manage my credit use \n \n *Saya memiliki pemahaman yang cukup dalam menggunakan kartu kredit*', options, key=f'question_23')
        q24 = container_4.radio(f' 24 . I have the ability to maintain financial records for my income and expenditure \n \n *Saya melakukan pencatatan keuangan saya terkait pemasukan dan pengeluaran*', options, key=f'question_24')
        q25 = container_4.radio(f'  25 . I can manage my money easily \n \n *Saya dapat mengatur keuangan dengan mudah*', options, key=f'question_25')
        q26 = container_4.radio(f'  26 . I have a better understanding of financial instruments (e.g. Bonds, stock, T-bill, time value of money, future contract, option, etc.) \n \n *Saya memiliki pemahaman yang cukup baik pada instrumen keuangan*', options, key=f'question_26')
        q27 = container_4.radio(f'  27 . I have the ability to prepare my own budget weekly and monthly \n \n *Saya memiliki kemampuan untuk membuat anggaran saya sendiri setiap minggu dan setiap bulan*', options, key=f'question_27')
        q28 = container_4.radio(f'  28 . I have the ability to prepare my own budget weekly and monthly \n \n *Saya memiliki kemampuan untuk membuat anggaran saya sendiri setiap minggu dan setiap bulan*', options, key=f'question_28')
        q29 = container_4.radio(f'  29 . I have the ability to prepare my own budget weekly and monthly \n \n *Saya memiliki kemampuan untuk membuat anggaran saya sendiri setiap minggu dan setiap bulan*', options, key=f'question_29')
        q30 = container_4.radio(f'  30 . I have the ability to prepare my own budget weekly and monthly \n \n *Saya memiliki kemampuan untuk membuat anggaran saya sendiri setiap minggu dan setiap bulan*', options, key=f'question_30')
        
        submit_button = st.form_submit_button(label='Submit', use_container_width=True)

    
    data_inf = {
    "gender": gender,
    "age": age,
    "allowance": allowance,
    "q1": q1,
    "q2": q2,
    "q3": q3,
    "q4": q4,
    "q5": q5,
    "q6": q6,
    "q7": q7,
    "q8": q8,
    "q9": q9,
    "q10": q10,
    "q11": q11,
    "q12": q12,
    "q13": q13,
    "q14": q14,
    "q15": q15,
    "q16": q16,
    "q17": q17,
    "q18": q18,
    "q19": q19,
    "q20": q20,
    "q21": q21,
    "q22": q22,
    "q23": q23,
    "q24": q24,
    "q25": q25,
    "q26": q26,
    "q27": q27,
    "q28": q28,
    "q29": q29,
    "q30": q30
    }

    options = ["Disagree", "Agree", "Neutral", "Strongly Agree", "Very Strongly Agree"]

    if submit_button:
        # mapping result
        data_inf.iloc[0:1, 0:1].apply(lambda x: 0 if x == "Female" else 1)
        data_inf.iloc[0:1, 1:2].apply(lambda x: 0 if 17 <= x < 21 else (1 if 21<= x < 26 else 2))
        data_inf.iloc[0:1, 2:3].apply(lambda x: 0 if x < 500000 else (1 if 500000 < x <= 1000000 else 2))
        data_inf.iloc[0:1,3:].apply(lambda x: 1 if x == "Disagree" else (2 if x == "Agree" else (3 if x == "Neutral" else (4 if x == "Strongly Agree" else "Very Strongly Agree"))))
        
        # adjusting columns
        data_inf[final_features]

        # pca

        # predict cluster

        # create description

        # predict probability of recommendation

        # get instrument

        # calculate result 

        # plot result

if __name__ == '__main__':
    run()