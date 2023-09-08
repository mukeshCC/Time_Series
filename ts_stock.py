import streamlit as st, pandas as pd, numpy as np , yfinance as yf
import matplotlib.pyplot as plt, plotly.express as px,plotly.graph_objects as go
import math
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error,mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
#from tensorflow import layers
from sklearn.preprocessing import MinMaxScaler


st.title("Time Seires on Stocks ")
stocks = ('hdfcbank.ns','icicibank.ns','sbin.ns','reliance.ns','infy.ns','tcs.ns','itc.ns','lt.ns','adanient.ns','sunpharma.ns')
ticker = st.sidebar.selectbox('select stocks',stocks)
start = st.sidebar.date_input("Start Date")
end = st.sidebar.date_input("End Date")

def decomposition(data, period):
    result_decom = seasonal_decompose(data, model="additive", period=period, extrapolate_trend='freq')
    trend = result_decom.trend
    season = result_decom.seasonal
    resid = result_decom.resid
    return trend, season, resid

# Function for ACF and PACF plots
def acf_pacf_plot(data, lags):
    fig_acf = plot_acf(data, lags=lags)
    fig_pacf = plot_pacf(data, lags=lags)
    return fig_acf, fig_pacf

# Function for ADF test and displaying results
def adfuller_test(Close):
    adfuller_result = adfuller(Close, autolag='AIC')
    adfuller_output = pd.Series(adfuller_result[:4], index=['Test statistic', 'p-value',
                                                           'Lags Used','Number of Observations Used'])
    return adfuller_output

#forecast_steps = st.number_input("Enter th enumber of forecast steps ", len(test_data))

# def adfuller_test(df):
#     adfuller_result = adfuller(df, autolag='AIC')
#     adfuller_output = pd.Series(adfuller_result[:4], index=['Test statistic', 'p-value',
#                                                            'Lags Used','Number of Observations Used'])
#     return adfuller_output

#if ticker and start and end :
data = yf.download(ticker,start=start,end=end)
data = pd.DataFrame(data)
data

#tab1,tab2,tab3 = st.tabs("Data Visualization","Time Series Test","Arima & LSTM")
tabs = st.sidebar.radio("Select a Tab", ["Data Visualization", "Seasonal Decompose & ADF", "AutoARIMA","LSTM"])
#with tab1:
if tabs == "Data Visualization":
     fig = px.line(data,y='Close',title=ticker)
     fig

     fig_i = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close']),
                        ])
     fig_i.layout.update(xaxis_rangeslider_visible=True)
     st.title("CandleStick Chart")
     st.plotly_chart(fig_i)
          
    # Calculate daily returns
     data['Daily Returns'] = data['Adj Close'].pct_change()

     # Calculate monthly returns
     monthly_returns = data['Adj Close'].resample('M').ffill().pct_change()
     monthly_returns = monthly_returns*100

     quarterly_returns = data['Adj Close'].resample('Q').ffill().pct_change()
     quarterly_returns = quarterly_returns*100

     yearly_returns = data['Adj Close'].resample('Y').ffill().pct_change()
     yearly_returns = yearly_returns*100



     # Plot stock price and returns
     #st.subheader(f"Stock Price and Daily Returns for {ticker}")
     #st.line_chart(data[['Adj Close', 'Daily Returns']])

    # Plot monthly returns
     st.subheader(f"Monthly Returns for {ticker}")
     st.bar_chart(monthly_returns.dropna())
     st.subheader(f"Quarterly Returns for {ticker}")
     st.bar_chart(quarterly_returns.dropna())
     st.subheader(f"Yearly Returns for {ticker}")
     st.bar_chart(yearly_returns.dropna())

    #  # Combine returns into a single DataFrame for visualization
    #  combined_returns = pd.concat([monthly_returns, quarterly_returns, yearly_returns], axis=1)
    #  combined_returns.columns = ['Monthly Returns', 'Quarterly Returns', 'Yearly Returns']

    # # Plot combined returns
    #  st.subheader(f"Combined Returns for {ticker}")
    #  st.bar_chart(combined_returns.dropna())

if tabs == "Seasonal Decompose & ADF":
     st.title("Seasonal Decomposition of Stock Price Data")


     # Input for seasonal decomposition period
     period = st.number_input("Enter the seasonal decomposition period:", min_value=1)
      

     if st.button("Decompose"):
            # Perform seasonal decomposition
            trend, season, resid = decomposition(data['Close'], period)
            # # Calculate moving averages
            # moving_average_window = 10  # Adjust the window size as needed
            # moving_average_data = data['Close'].rolling(window=moving_average_window).mean()
            # moving_average_trend = trend.rolling(window=moving_average_window).mean()

            # Display the decomposed components
            st.subheader("Decomposed Components")
            st.line_chart(trend, use_container_width=True)
            st.write("Trend")

            st.line_chart(season, use_container_width=True)
            st.write("Seasonal")

            st.line_chart(resid, use_container_width=True)
            st.write("Residual")

            # Plot the decomposed components
            st.subheader("Decomposition Plot")
            fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(10, 6))
            result_decom = seasonal_decompose(data['Close'], model="additive", period=period, extrapolate_trend='freq')
            # Define result_decom here

            result_decom.trend.plot(ax=axes[0])
            axes[0].set_title("Trend")

            result_decom.seasonal.plot(ax=axes[1])
            axes[1].set_title("Seasonal")

            result_decom.resid.plot(ax=axes[2])
            axes[2].set_title("Residual")

            # Add a custom legend
            for ax in axes:
                ax.legend(["Component"])
            plt.tight_layout()
            st.pyplot(fig)
           

            # # Optionally, display summary statistics or other relevant information
            # col1,col2,col3 = st.columns(3)

            # col1.metric("Trend Summary",str(trend.describe()))
            # col2.metric('Seasonal Summary',str(season.describe()))
            # col3.metric("Residual Summary",str(resid.describe()))

            st.title("ACF and PACF Plots for Stock Price Data")

            # Input for the number of lags
            lags = 40
             # Generate and display ACF and PACF plots
            st.subheader("ACF and PACF Plots")
            acf_plot, pacf_plot = acf_pacf_plot(data['Close'], lags)

            st.pyplot(acf_plot)
            st.pyplot(pacf_plot)

              # Perform ADF test and display results
            st.subheader("ADF Test Results")
            adf_result = adfuller_test(data['Close'])
            st.write(adf_result)

            differencing = 1
            # Input for differencing
            #differencing = st.number_input("Enter the differencing order:", min_value=0)
            differenced_data = data['Close'].diff(differencing).dropna()

            # Perform seasonal decomposition on differenced data
            trend, season, resid = decomposition(differenced_data, period)

            # Display the decomposed components of the differenced data
            st.subheader("Decomposed Components (Differenced Data)")
            
            st.line_chart(trend, use_container_width=True)
            st.write("Trend")

            st.line_chart(season, use_container_width=True)
            st.write("Seasonal")

            st.line_chart(resid, use_container_width=True)
            st.write("Residual")

            # Plot the decomposed components with a custom legend
            st.subheader("Decomposition Plot (Differenced Data)")
            fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(10, 6))
            
            result_decom = seasonal_decompose(differenced_data, model="additive", period=period, extrapolate_trend='freq')
            
            result_decom.trend.plot(ax=axes[0])
            axes[0].set_title("Trend")

            result_decom.seasonal.plot(ax=axes[1])
            axes[1].set_title("Seasonal")

            result_decom.resid.plot(ax=axes[2])
            axes[2].set_title("Residual")

            # Add a custom legend
            for ax in axes:
                ax.legend(["Component"])
            plt.tight_layout()
            st.pyplot(fig)

            # Generate and display ACF and PACF plots for differenced data
            st.subheader("ACF and PACF Plots (Differenced Data)")
            acf_plot, pacf_plot = acf_pacf_plot(differenced_data, lags)

            st.pyplot(acf_plot)
            st.pyplot(pacf_plot)

            # Perform ADF test on differenced data and display results
            st.subheader("ADF Test Results (Differenced Data)")
            adf_result = adfuller_test(differenced_data)
            st.write(adf_result)
    
            # lag = 30
            # ## Generate and display ACF and PACF plots for differenced data
            # st.subheader("ACF and PACF Plots (Differenced Data)")
            # acf_plot, pacf_plot = acf_pacf_plot(data['diff_data'], lag)

            # st.pyplot(acf_plot)
            # st.pyplot(pacf_plot)

            # # Perform ADF test on differenced data and display results
            # st.subheader("ADF Test Results (Differenced Data)")
            # adf_result1 = adfuller_test(data['diff_data'].dropna())
            # st.write(adf_result1)

#with tab3:
if tabs == "AutoARIMA":
            st.subheader("Auto_Arima")

            train_ratio = st.slider("Train Data Ratio (%)", 1, 99, 80)
            test_ratio = 100 - train_ratio

            # Calculate the split index
            split_index = int(len(data) * train_ratio / 100)

            # Split the data into train and test sets
            train_data = data.iloc[:split_index]
            test_data = data.iloc[split_index:]

            # Plot the data
            plt.figure(figsize=(10, 6))
            plt.grid(True)
            plt.xlabel('Date')
            plt.ylabel('Closing Price')
            plt.plot(train_data.index, train_data['Close'], 'green', label='Train data')
            plt.plot(test_data.index, test_data['Close'], 'blue', label='Test data')
            plt.title(f'{ticker} Train and Test Data')
            plt.legend()

            # Display the plot
            st.pyplot(plt)

            # Fit the autoARIMA model
            model_autoARIMA = auto_arima(train_data['Close'], start_p=0, start_q=0,
                                        test='adf', max_p=3, max_q=3,
                                        m=1, d=None, seasonal=False,
                                        start_P=0, D=0, trace=True,
                                        error_action='ignore',
                                        suppress_warnings=True,
                                        stepwise=True)

            # Display model summary
            st.subheader("AutoARIMA Model Summary")
            st.text(str(model_autoARIMA.summary()))

            # Plot model diagnostics
            st.subheader("AutoARIMA Model Diagnostics")
            model_autoARIMA.plot_diagnostics(figsize=(15, 8))
            st.pyplot(plt)

            # Fit an ARIMA model to the test data
            st.subheader("ARIMA Model and Forecast")
            p, d, q = model_autoARIMA.order
            model = ARIMA(train_data['Close'], order=(p, d, q))
            fitted = model.fit()

            # Display the model summary
            st.subheader("ARIMA Model Summary")
            st.text(str(fitted.summary()))

            
            # Forecast future values
            forecast_steps = len(test_data)  # Forecast the same number of steps as test data
            forecast = fitted.forecast(steps=forecast_steps, alpha=0.05)

            # Create forecast dates
            forecast_dates = pd.date_range(start=test_data.index[-1], periods=forecast_steps + 1)
            # Create a DataFrame for the forecasted data with the same date range as test_data
            forecast_df = pd.DataFrame({'Close': forecast}, index=forecast_dates)

            # Concatenate the test data and forecasted data
            combined_data = pd.concat([test_data, forecast_df])

            #Plot the forecast and actual test data
            plt.figure(figsize=(10, 6))
            plt.grid(True)
            plt.xlabel('Date')
            plt.ylabel('Closing Price')
            plt.plot(train_data.index, train_data['Close'], 'green', label='Train data')
            #plt.plot(test_data.index, test_data['Close'], 'blue', label='Test data')
            #plt.plot(forecast_dates[1:], forecast, 'orange', label='Forecasted data')
            #plt.plot(combined_data.index, combined_data['Close'], 'blue', label='Test data and Forecasted data')
            plt.plot(combined_data.index, combined_data['Close'], 'b-', label='Test data', linestyle='-')
            plt.plot(forecast_df.index, forecast_df['Close'], 'orange', label='Forecasted data')
            plt.title(f'{ticker} ARIMA Model Forecast')
            plt.legend()
            st.pyplot(plt)
             # Create a DataFrame to combine the date range, actual test data, and forecasted values
            forecast_df = pd.DataFrame({
                'Actual Stock Price': test_data['Close'].values,
                'Forecasted Stock Price': forecast
            })

            # # Print the DataFrame
            st.subheader("Actual vs. Forecasted Stock Prices")
            st.write(forecast_df)

            # #Forecast future values
            #forecast_steps = st.write("number of forecast steps:", len(test_data))
            # #forecast_steps = st.number_input("Enter the number of forecast steps:", min_value=1)
            # forecast = fitted.forecast(steps=forecast_steps, alpha=0.05)
            #forecast_dates = pd.date_range(start=test_data.index[-1], periods=forecast_steps)
            # forecast_dates = pd.date_range(start=test_data.index[-1], periods=forecast_steps + 1, freq='D')[1:]

            # if len(forecast) > len(test_data):
            #         forecast = forecast[:len(test_data)]

            #     # Plot the forecast and actual test data
            # plt.figure(figsize=(10, 5), dpi=100)
            # plt.plot(test_data.index, test_data['Close'], label='Actual Stock Price', color='blue')
            # plt.plot(forecast_dates, forecast, label='Predicted Stock Price', color='orange')
            # plt.title(f'{ticker} Stock Price Comparison')
            # plt.xlabel('Time')
            # plt.ylabel('Stock Price')
            # plt.legend()
            # st.pyplot(plt)

            # #Plot the forecast
            # plt.figure(figsize=(10, 5), dpi=100)
            # plt.plot(train_data.index, train_data['Close'], label='Training data')
            # plt.plot(test_data.index, test_data['Close'], color='blue', label='Actual Stock Price')
            # plt.plot(forecast_dates, forecast, color='orange', label='Predicted Stock Price')
            # #plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=0.1)
            # plt.title(f'{ticker} Stock Price Prediction')
            # plt.xlabel('Time')
            # plt.ylabel('Stock Price')
            # plt.legend()
            # st.pyplot(plt)

                # Calculate and display performance metrics
            mse = mean_squared_error(test_data['Close'], forecast)
            mae = mean_absolute_error(test_data['Close'], forecast)
            rmse = math.sqrt(mse)
            #mape = np.mean(np.abs(forecast - test_data['Close']) / np.abs(test_data['Close']))
            mape = np.mean(np.abs((test_data['Close'] - forecast) / test_data['Close'])) * 100
            # Calculate accuracy score
            accuracy = 100 - mape

            st.subheader("Performance Metrics")
            st.write(f'MSE: {mse}')
            st.write(f'MAE: {mae}')
            st.write(f'RMSE: {rmse}')
            st.write(f'MAPE: {mape}')
            st.write(f'Accuracy: {accuracy:.2f}%')

          
if tabs == "LSTM":
            st.subheader(" Using LSTM Stock Forecasting")
                # Input for LSTM hyperparameters
            look_back = st.slider("Look-back Period", min_value=1, max_value=30, value=10, step=1)
            num_epochs = st.slider("Number of Epochs", min_value=1, max_value=100, value=50, step=1)
            batch_size = st.slider("Batch Size", min_value=1, max_value=128, value=32, step=1)

            # Preprocess the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

            # Create sequences of data for LSTM training
            X, y = [], []
            for i in range(look_back, len(scaled_data)):
                X.append(scaled_data[i - look_back:i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)

            # Split the data into train and test sets
            split_ratio = 0.8
            split_index = int(len(X) * split_ratio)
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]

            # Define and compile the LSTM model
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(units=50, activation='relu', input_shape=(look_back, 1)),
                tf.keras.layers.Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the LSTM model
            model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

            # Make predictions using the LSTM model
            predicted_stock_prices = model.predict(X_test)
            
            # Inverse transform the predictions to original scale
            predicted_stock_prices = scaler.inverse_transform(predicted_stock_prices)
            y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

            # Create a date range for the test data
            test_dates = data.index[-len(y_test):]

            # Plot the LSTM predictions and actual test data
            plt.figure(figsize=(10, 5), dpi=100)
            plt.plot(test_dates, y_test, label='Actual Stock Price', color='blue')
            plt.plot(test_dates, predicted_stock_prices, label='LSTM Predicted Stock Price', color='green')
            plt.title(f'{stocks} Stock Price Comparison (LSTM)')
            plt.xlabel('Time')
            plt.ylabel('Stock Price')
            plt.legend()
            st.pyplot(plt)

                        # Calculate performance metrics
            mse = mean_squared_error(y_test, predicted_stock_prices)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predicted_stock_prices)
            mape = np.mean(np.abs((y_test - predicted_stock_prices) / y_test)) * 100
            accuracy = 100 - mape

            comparison_df = pd.DataFrame({
                 'Actual Stock Price': y_test.flatten(),
                 'LSTM Predicted Stock Price': predicted_stock_prices.flatten()})
            
            
            # Display the comparison DataFrame
            st.subheader("Comparison between Actual and Predicted Stock Prices")
            st.dataframe(comparison_df)


            # Display performance metrics
            st.subheader("Performance Metrics")
            st.write(f"MSE: {mse:.2f}")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"MAPE: {mape:.2f}%")
            st.write(f"Accuracy: {accuracy:.2f}%")

            fig = px.line(comparison_df, x=comparison_df.index, y=['Actual Stock Price', 'LSTM Predicted Stock Price'], title=f'{ticker} Stock Price Comparison (LSTM)')
            fig.update_xaxes(title='Time')
            fig.update_yaxes(title='Stock Price')
            st.plotly_chart(fig)



          

            # ... (plotting code remains the same)
            # In this code:

            # # After making predictions using the LSTM model, we calculate the following performance metrics:

            # MSE (Mean Squared Error)
            # RMSE (Root Mean Squared Error)
            # MAE (Mean Absolute Error)
            # MAPE (Mean Absolute Percentage Error)
            # Accuracy (calculated as 100 - MAPE)
# We display these performance metrics using Streamlit's st.write function under the "Performance Metrics" subheader.

# Now, when you run your Streamlit app, it will not only plot the LSTM predictions and actual test data but also display the performance metrics in the Streamlit interface. Users can assess the model's accuracy and error using these metrics.














            # selected_cloumn = 'Close'

            # ts_data = data[selected_cloumn]

            # # Fit auto ARIMA model
            # model = pm.auto_arima(ts_data, seasonal=True, m=period, stepwise=True, trace=True)
            # st.subheader("Auto ARIMA Model Summary")
            # st.write(model.summary())

            # # Forecast future values
            # forecast = model.predict(n_periods=forecast_steps)
            # st.subheader("Auto ARIMA Forecast")
            # st.write(forecast)

            # # Plot the forecast
            # plt.figure(figsize=(10, 6))
            # plt.plot(data['Close'], label="Original Data")
            # plt.plot(range(len(data), len(data) + forecast_steps), forecast, label="Forecast", linestyle='--')
            # plt.xlabel("Time")
            # plt.ylabel("Value")
            # plt.legend()
            # st.pyplot(plt)

            # # Plot the forecast with Plotly Express
            # fig = px.line(data_frame=ts_data, x=ts_data.index, y=ts_data[selected_cloumn], labels={"x": "Time", "y": "Value"},
            #   title="Original Data")
            # fig.add_trace(px.line(data_frame=forecast, x=forecast['Date'], y=forecast, line_dash="dash",
            #           labels={"x": "Time", "y": "Value"}, name="Forecast").data[0])
            # fig.update_layout(showlegend=True)








#from alpha_vantage.fundamentaldata import FundamentalData
#import requests

# # define simple function get all the information needed
# def information_func(data):
    
#     # unique stocks
#     print("Uniques stocks available in dataset:", data['Name'].nunique())
#     st.print("----"*20)
    
#     # metadata of dataset
#     print("Metadata of the dataset:\n")
#     data.info()
#     print("----"*20)
    
#     # missing values
#     null = data.isnull().sum()
#     print(null)
#     print("----"*20)
    
#     # max range of stocks dataset
#     delta = (pd.to_datetime(data['date']).max() - pd.to_datetime(data['date']).min())
#     print("Time range of stocks dataset:\n", delta)
#     print("----"*20)

# let's find seasonla decomposition of time-series models
# def decomposition(data, period):
#     # decompistion instance
#     result_decom = seasonal_decompose(data['Close'], model="additive", 
#                                       period=period, extrapolate_trend='freq')
#     # plot the componenets 
#     fig = result_decom.plot()
#     fig.set_size_inches((10, 6))
#     # Tight layout to realign things
#     fig.tight_layout()
#     plt.show()
    
#     # capture the compoenets 
#     trend = result_decom.trend
#     season = result_decom.seasonal
#     reside = result_decom.resid
#     return trend, season, reside





