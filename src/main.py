import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np

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

def closed_form_solution(df, x_label, y_label):
    x = df[x_label]
    y = df[y_label]
    n = len(df)
    
    # Solve for m
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x_squared = (x ** 2).sum()
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)

    # Solve for b
    b = (sum_y - m * sum_x) / n

    return m, b

def correlation_coefficient(df, x_label, y_label):
    x = df[x_label]
    y = df[y_label]
    correlation_matrix = np.corrcoef(x, y)
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient


def plot_chart(x_label, y_label, df, slope, intercept, x_lim=None, y_lim=None):
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

    if x_lim is not None:
        plt.xlim(x_lim[0], x_lim[1])

    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])

    st.pyplot(plt)


def load_data():
    # Load data
    data_path = os.path.join(os.getcwd(), 'data', 'realestate.csv') 
    columns = ['TransactionDate', 'HouseAge', 'DistanceToNearestMRTStation', 'NumberConvenienceStores', 'Latitude', 'Longitude', 'PriceOfUnitArea']
    df = pd.read_csv(data_path, names=columns)
    preprocess(df)

    print(df['TransactionDate'].min())
    print(df['TransactionDate'].max())

    return df

def preprocess(df):
    # Make the transaction date as years only by removing the digits after the decimal point
    df['TransactionDate'] = df['TransactionDate'].astype(str).str.split('.').str[0].astype(int)
    return df

def reset():
    st.session_state.slope = 1.0
    st.session_state.intercept = 1.0

def compute():
    st.session_state.slope, st.session_state.intercept = closed_form_solution(df, features, target)

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
    slope = st.slider('Slope', min_value=-50.0, max_value=50.0, step=0.1, key='slope')
    intercept = st.slider('Intercept', min_value=0.0, max_value=100.0, step=0.1, key='intercept')

    # Write the cost function (mean squared error)
    sum = sum_of_squared_residuals(df, features, target, slope, intercept)
    n = len(df)
    cost = mean_squared_error(sum, n)
    st.markdown(f'#### Cost: {cost}')

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
    reset = col1.button('Reset', on_click=reset)
    calculate = col2.button('Calculate', on_click=compute)

    if calculate:
        st.markdown(f'#### Slope: {st.session_state.slope}')
        st.markdown(f'#### Intercept: {st.session_state.intercept}')
        st.markdown(f'#### Correlation Coefficient: {correlation_coefficient(df, features, target)}')
        
    # Plot chart
    plot_chart(features, target, df, slope, intercept)