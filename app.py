import numpy as np
import streamlit as st
import pickle

# Page Configuration
st.set_page_config(page_title="Credit Loan Prediction", page_icon=":money_with_wings:", layout="wide")

# Custom Background
page_bg_color = '''
<style>
body {
    background-color: #F0F8FF;
}
</style>
'''
st.markdown(page_bg_color, unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: blue;'>Credit Loan Prediction</h1>", unsafe_allow_html=True)

# Load Model
with open('Models/model_XGB.pkl', 'rb') as file:
    model = pickle.load(file)

def label_input(input):
    ownership_list = ['MORTGAGE','OTHER','OWN','RENT']
    intent_list = ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
    grade_list = ['A','B','C','D','E','F','G']
    hc_default_list = ['No', 'Yes']

    input[1] = int(ownership_list.index(input[1]))
    input[3] = int(intent_list.index(input[3]))
    input[4] = int(grade_list.index(input[4]))
    input[8] = int(hc_default_list.index(input[8]))

    return input

def prediction(input):
    Final_input = label_input(input)
    Final_input = np.array(Final_input).reshape(1, -1)  # Reshape to 2D array (1 sample, multiple features)
    pred = model.predict(Final_input)
    return 'Default' if pred[0] == 1 else 'Not Default'

def main():
    st.markdown("## Welcome to the Credit Loan Prediction App")
    st.markdown("This app helps you predict the likelihood of a loan defaulting based on user inputs.")

   
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=20, max_value=80, value=30)
    with col2:
        emp_length = st.slider("Employed since last (years):", min_value=0, max_value=30, value=25)


    income = st.slider("Income:", min_value=1000, max_value=1000000, value=400000, step=1000)
    amount = st.slider("Loan Amount:", min_value=500, max_value=35000, value=30000, step=100)
    rate = st.slider("Interest Rate:", min_value=1.0, max_value=30.0, value=10.0, step=0.1)
    ch_length = st.slider("Credit History Length:", min_value=0, max_value=50, value=40, step=1)

    col3, col4, col5, col6 = st.columns(4)

    with col3:
        ownership = st.selectbox("Home Ownership", ['MORTGAGE','OTHER','OWN','RENT'])
    with col4:
        intent = st.selectbox("Loan Intent", ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'])
    with col5:
        grade = st.selectbox("Loan Grade", ['A','B','C','D','E','F','G'])
    with col6:
        hc_default = st.radio("Historical Default", ('No','Yes'))
        
    lp_income = (amount*100)/income   # Loan Amount percent with respect to income

    input = [age, ownership, emp_length, intent, grade, amount, float(rate), float(lp_income), hc_default, ch_length]

    button_style = """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    if st.button("Check Result"):
        result = prediction(input)
        st.markdown(f'<h3 style="color:green;">{result}</h3>', unsafe_allow_html=True)




if __name__ == '__main__':
    main()
