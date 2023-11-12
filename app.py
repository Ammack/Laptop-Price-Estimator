import streamlit as st
import pandas as pd
import numpy as np
import math
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# load the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

data = pd.read_pickle(open("data.pkl", "rb"))

st.title('Laptop Price Predictor')

# Company
company = st.selectbox('Brand',data['Company'].unique())

# type of laptop
type = st.selectbox('Type',data['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop (in Kg))')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
IPS = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input('Screen Size (in inches)')

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU Brand',data['Cpu Name'].unique())

# cpu_speed = st.number_input('CPU speed (in GHz)')

hdd = st.selectbox('HDD (in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD (in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',data['Gpu Brand'].unique())

OpSys = st.selectbox('Operating System',data['OpSys'].unique())

if st.button('Predict Price'):
    #preprocessing
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if IPS == 'Yes':
        IPS = 1
    else:
        IPS = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    values = pd.DataFrame([[company,type,ram,OpSys,weight,touchscreen,IPS,ppi,cpu,2.5,hdd,ssd,gpu]],
                        columns=['Company','TypeName','Ram','OpSys','Weight','Touchscreen','IPS','PPI','Cpu Name','Cpu Speed','HDD','SSD','Gpu Brand'])

    prediction = pipe.predict(values)

    st.text(f"The predicted price of this configuration is between {round(int(np.exp(prediction[0])),-3)} to {round(int(np.exp(prediction[0])+1000),-3)}")
