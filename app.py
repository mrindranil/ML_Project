import streamlit as st
import pickle

st.title('ML Project')
displacement = st.number_input('Displacement', value=300, placeholder='Enter a value')
horsepower = st.number_input('Horsepower', value=150, placeholder='Enter a value')
weight = st.number_input('Weight', value=5000, placeholder='Enter a value')
accelaration = st.number_input('Accelaration', value=12, placeholder='Enter a value')

loaded_model = pickle.load(open('mpg_regression.sav', 'rb'))

prediction=loaded_model.predict([[displacement,horsepower,weight,accelaration]])
st.subheader(f'predicted mpg value for above parameter is{prediction[0]}')
st.write(displacement,horsepower,weight,accelaration)