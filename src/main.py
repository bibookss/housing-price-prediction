import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os

def find_y(x, m, b):
    y = m * x + b
    return y

def sum_of_squared_residuals(df, x_label, y_label, slope, intercept):
    sum = 0
    for index, row in df.iterrows():
        y = find_y(row[x_label], slope, intercept)
        sum += (y - row[y_label]) ** 2
    return sum

def mean_squared_error(sum, n):
    return sum / n

def plot_chart(x_label, y_label, df, slope, intercept):
    plt.figure(figsize=(8, 6))
    plt.title(f'{y_label} vs {x_label}')
    sns.scatterplot(x=x_label, y=y_label, data=df, s=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    lineStart = -5
    lineEnd = df[x_label].max()

    x = [lineStart, lineEnd]
    y = [find_y(lineStart, slope, intercept), find_y(lineEnd, slope, intercept)]

    plt.plot(x, y, 'r-', linewidth=2)

    plt.xlim(lineStart, lineEnd)
    plt.ylim(-5, df[y_label].max())

    st.pyplot(plt)

def load_data():
    # Load data
    data_path = os.path.join(os.getcwd(), 'data', 'realestate.csv') 
    columns = ['TransactionDate', 'HouseAge', 'DistanceToNearestMRTStation', 'NumberConvenienceStores', 'Latitude', 'Longitude', 'PriceOfUnitArea']
    df = pd.read_csv(data_path, names=columns)
    preprocess(df)

    return df

def preprocess(df):
    # Make the transaction date as years only by removing the digits after the decimal point
    df['TransactionDate'] = df['TransactionDate'].astype(str).str.split('.').str[0].astype(int)
    return df

if __name__ == '__main__':
    # Load data
    df = load_data()

    # Sidebar
    st.sidebar.header('Features')
    st.sidebar.markdown('Select the features you want to plot')
    features = st.sidebar.selectbox('Features', df.columns, index=1)
    target = st.sidebar.selectbox('Target', df.columns, index=6)

    # Plot chart
    st.title('Real Estate Price Prediction')
    st.subheader('Using Univariate Linear Regression')

    # Get user input by slider for slope and intercept
    slope = st.slider('Slope', min_value=-50.0, max_value=50.0, value=1.0, step=0.1)
    intercept = st.slider('Intercept', min_value=0.0, max_value=100.0, value=1.0, step=0.1)

    # Write the cost function (mean squared error)
    sum = sum_of_squared_residuals(df, features, target, slope, intercept)
    n = len(df)
    cost = mean_squared_error(sum, n)
    st.markdown(f'Cost: {cost}')

    # Add reset and calculate button
    st.markdown("""
            <style>
                div[data-testid="column"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="column"] * {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)

    col1, col2 = st.columns([1,1])
    reset = col1.button('Reset')
    calculate = col2.button('Calculate')

    if reset:
        st.write('Resetting...')
        slope = 1.0
        intercept = 1.0

    if calculate:
        st.write('Calculating...')
        
    # Plot chart
    plot_chart(features, target, df, slope, intercept)