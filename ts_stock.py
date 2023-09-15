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
import pmdarima as pm



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
# def adfuller_test(Close):
#     adfuller_result = adfuller(Close, autolag='AIC')
#     adfuller_output = pd.Series(adfuller_result[:4], index=['Test statistic', 'p-value',
#                                                            'Lags Used','Number of Observations Used'])
#     return adfuller_output

def adfuller_test(data):
    result = adfuller(data)
    st.write('ADF Statistic:', result[0])
    st.write('p-value:', result[1])
    st.write('Critical Values:', result[4])
    if result[1] <= 0.05:
        st.write('Stationary (Reject H0)')
    else:
        st.write('Non-Stationary (Fail to Reject H0)')


# # Create a dictionary to store RMSE values in their respective tabs
# rmse_values = {
#     "ARIMA": None,
#     "AutoARIMA": None,
#     "SARIMA": None,
#     "LSTM": None,
# }


#forecast_steps = st.number_input("Enter th enumber of forecast steps ", len(test_data))

# def adfuller_test(df):
#     adfuller_result = adfuller(df, autolag='AIC')
#     adfuller_output = pd.Series(adfuller_result[:4], index=['Test statistic', 'p-value',
#                                                            'Lags Used','Number of Observations Used'])
#     return adfuller_output

#if ticker and start and end :
data = yf.download(ticker,start=start,end=end)
data = pd.DataFrame(data)
#data
#tab1,tab2,tab3 = st.tabs("Data Visualization","Time Series Test","Arima & LSTM")
tabs = st.sidebar.radio("Select a Tab", ["Data Visualization", "Seasonal Decompose & ADF",
                                         "ARIMA and Auto ARIMA","Weekly",
                                         "SARIMA on Weekly Data","LSTM","Conclusion"])
#"Monthly","Monthly Arima","Monthly Analysis"
# Add an input field for differencing order

#with tab1:
if tabs == "Data Visualization":
     st.title("Time Seires on Stocks ")

     st.subheader("Project Objective:")
     st.markdown("Perform in-depth time series analysis on historical stock price data "
                " explore various forecasting models, and evaluate their accuracy to enable informed investment decisions.")
     
     st.title("Dataset")
     data
     fig = px.line(data,y='Close',title=ticker)
     fig

     fig_i = go.Figure(data=[go.Candlestick(x=data.index,
                    open=data['Open'], high=data['High'],
                    low=data['Low'], close=data['Close']),
                        ])
     fig_i.layout.update(xaxis_rangeslider_visible=False)
    #  fig_i.update_layout(
    #       title=f"{ticker} Stock Price",
    #       xaxis_rangeslider_visible=True,
    #       xaxis_rangeslider=dict(
    #            visible=True,
    #            rangeselector=dict(
    #            buttons=list([
    #                dict(count=1, label="1m", step="month", stepmode="backward"),
    #                dict(count=6, label="6m", step="month", stepmode="backward"),
    #                dict(count=1, label="YTD", step="year", stepmode="todate"),
    #                dict(count=1, label="1y", step="year", stepmode="backward"),
    #                dict(step="all")])
    #                )))
     st.title("CandleStick Chart")
     st.plotly_chart(fig_i)
          
    # Calculate daily returns
     #returns = st.selectbox("monthly_returns","quarterly_returns","yearly_returns")
     data['Daily Returns'] = data['Adj Close'].pct_change()
     
     # Calculate monthly returns
     monthly_returns = data['Adj Close'].resample('M').ffill().pct_change()*100
     #monthly_returns = monthly_returns*100

     quarterly_returns = data['Adj Close'].resample('Q').ffill().pct_change()*100
     #quarterly_returns = quarterly_returns*100

     yearly_returns = data['Adj Close'].resample('Y').ffill().pct_change()*100
     #yearly_returns = yearly_returns*100
    
     returns = {'monthly':monthly_returns,'quarterly':quarterly_returns,'yearly':yearly_returns}

     st.title("Stock return Analysis")
     selected_option = st.selectbox("Select Returns Range",list(returns.keys()))

     if selected_option in returns:
           st.subheader(f"{selected_option} Returns for {ticker}")
           st.write(returns[selected_option])
           st.bar_chart(returns[selected_option].dropna())





     # Plot stock price and returns
     #st.subheader(f"Stock Price and Daily Returns for {ticker}")
     #st.line_chart(data[['Adj Close', 'Daily Returns']])



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

     differencing = st.number_input("Enter the differencing order:", min_value=0)
      # Set the global variable
     
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

            #differencing = 1
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

# Create a new tab for ARIMA and Auto ARIMA
if tabs == "ARIMA and Auto ARIMA":
    st.title("ARIMA and Auto ARIMA Analysis")
    # Input for ARIMA parameters
    p_value = st.number_input("Enter the p value (order of AR):", min_value=0)
    d_value = st.number_input("Enter the d value (order of differencing):", min_value=0)
    q_value = st.number_input("Enter the q value (order of MA):", min_value=0)
    train_test_ratio = st.slider("Select Train-Test Split Ratio:", 0.1, 0.9, 0.7)

    if st.button("Run ARIMA Model"):
        # Split data into train and test sets
        train_size = int(len(data) * train_test_ratio)
        train_data, test_data = data['Close'][:train_size], data['Close'][train_size:]

         # Plot the data
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.plot(train_data.index, train_data, 'green', label='Train data')
        plt.plot(test_data.index, test_data, 'blue', label='Test data')
        plt.title(f'{ticker} Train and Test Data')
        plt.legend()

        # Display the plot
        st.pyplot(plt)

        # Fit ARIMA model
        model = ARIMA(train_data, order=(p_value, d_value, q_value))
        model_fit = model.fit()

        # Make forecasts
        forecast = model_fit.forecast(steps=len(test_data))

            # Display ARIMA model summary
        st.subheader("ARIMA Model Summary")
        st.text(model_fit.summary())

        # # Create a DataFrame to display test data and forecasted data
        # result_df = pd.DataFrame({'Test Data': test_data, 'Forecasted Data': forecast})
        # st.subheader("Test Data vs. Forecasted Data")
        # st.write(result_df)

        # Calculate evaluation metrics
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        rmse = math.sqrt(mse)

        # Plot actual vs. forecasted values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_data.index, test_data.values, label="Actual")
        ax.plot(test_data.index, forecast, label="Forecast", color='orange')
        #ax.fill_between(test_data.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.5, label="Prediction Interval")
        ax.legend()
        st.subheader("ARIMA Forecast vs. Actual")
        st.pyplot(fig)

        st.subheader("ARIMA Model Evaluation")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    if st.button("Run Auto ARIMA Model"):
        # Split data into train and test sets
        #data1 = data['Close']
        train_size = int(len(data) * train_test_ratio)
        train_data, test_data = data['Close'][:train_size], data['Close'][train_size:]

         # Plot the data
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.plot(train_data.index, train_data, 'green', label='Train data')
        plt.plot(test_data.index, test_data, 'blue', label='Test data')
        plt.title(f'{ticker} Train and Test Data')
        plt.legend()

                  
        model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                        test='adf', max_p=5, max_q=5,
                        m=1, d=None, seasonal=False,
                        start_P=0, D=0, trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True)
        # Print the model summary
        st.subheader("Auto ARIMA Model Summary")
        st.text(model_autoARIMA.summary())

    # Make forecasts for the test data
        forecast= model_autoARIMA.predict(n_periods=len(test_data))
        #forecast conf_int = model_autoARIMA.predict(n_periods=len(test_data), return_conf_int=True)

        # Convert date index to a list of strings for x-axis
        # date_strings = [str(date) for date in test_data.index]
        # test_data['Date'] = date_strings
        #     # Convert DataFrame columns to NumPy arrays
        # test_data = test_data.to_numpy()
        # forecast = forecast.ravel()
        
    # Calculate evaluation metrics
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        rmse = math.sqrt(mse)
        st.subheader("Auto ARIMA Model Evaluation")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Create a DataFrame to display test data, forecasted data, and prediction intervals
        #result_df = pd.DataFrame({'Test Data': test_data, 'Forecasted Data': forecast})
        # result_df['Lower Bound'] = conf_int[:, 0]
        # result_df['Upper Bound'] = conf_int[:, 1]
        #st.subheader("Test Data vs. Forecasted Data (with Prediction Intervals)")
        #st.write(result_df)

        # Plot actual vs. forecasted values
        fig, ax = plt.subplots(figsize=(10, 6))
        #x_axis = range(len(test_data))
        ax.plot(test_data.index, test_data, label="Actual")
        ax.plot(test_data.index, forecast, label="Forecast", color='orange')
        #ax.fill_between(date_strings, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.5, label="Prediction Interval")
        ax.legend()
        #ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        #plt.xticks(rotation=45) 
        st.subheader("Auto ARIMA Forecast vs. Actual")
        st.pyplot(fig)


if tabs == "Weekly":
            #Resample daily data to monthly frequency (taking the mean)
            monthly_data = data['Adj Close'].resample('W').mean()
            monthly_data = pd.DataFrame(monthly_data)
            monthly_data

            period = st.number_input("Enter the seasonal decomposition period:", min_value=1)

            differencing = st.number_input("Enter the differencing order:", min_value=0)


            # Perform seasonal decomposition on monthly data
            result_decom = seasonal_decompose(monthly_data, model="additive",period=period)

            # Plot decomposed components
            st.subheader("Decomposed Components")
            fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(monthly_data.index, monthly_data, label='Original', color='blue')
            axes[0].set_ylabel('Original')
            axes[1].plot(monthly_data.index, result_decom.trend, label='Trend', color='green')
            axes[1].set_ylabel('Trend')
            axes[2].plot(monthly_data.index, result_decom.seasonal, label='Seasonal', color='red')
            axes[2].set_ylabel('Seasonal')
            axes[3].plot(monthly_data.index, result_decom.resid, label='Residual', color='purple')
            axes[3].set_ylabel('Residual')
            st.pyplot(fig)

            # Interpret the results and provide insights
            st.subheader("Insights")
            if any(abs(result_decom.seasonal) > 0.1):  # You can adjust the threshold for seasonality detection
                st.write("The weekly data appears to have a seasonal component.")
            else:
                st.write("No clear seasonal pattern detected in the weekly data.")

            st.title("ACF and PACF Plots for Stock Price Data")

            # Input for the number of lags
            lags = 40
             # Generate and display ACF and PACF plots
            st.subheader("ACF and PACF Plots")
            acf_plot, pacf_plot = acf_pacf_plot(monthly_data, lags)

            st.pyplot(acf_plot)
            st.pyplot(pacf_plot)

              # Perform ADF test and display results
            st.subheader("ADF Test Results")
            adf_result = adfuller_test(monthly_data)
            st.write(adf_result)

            # differencing = 1
            # Input for differencing
            #differencing = st.number_input("Enter the differencing order:", min_value=0)
            differenced_data = monthly_data.diff(differencing).dropna()
            # Perform seasonal decomposition on differenced data
           
            # trend, season, resid = decomposition(differenced_data,period=period)
            # # Display the decomposed components of the differenced data
            # st.subheader("Decomposed Components (Differenced Data)")
            
            # st.write("Trend")
            # st.line_chart(trend, use_container_width=True)
            # st.write("Seasonal")
            # st.line_chart(season, use_container_width=True)
            # st.write("Residual")
            # st.line_chart(resid, use_container_width=True)
            
            # Plot the decomposed components with a custom legend
            st.subheader("Decomposition Plot (Differenced Data)")
            fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(10, 6))
            
            result_decom = seasonal_decompose(differenced_data, model="additive",period=period)
            
            result_decom.trend.plot(ax=axes[0],color="green")
            axes[0].set_title("Trend")

            result_decom.seasonal.plot(ax=axes[1],color="red")
            axes[1].set_title("Seasonal")

            result_decom.resid.plot(ax=axes[2],color="purple")
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

            st.write(len(monthly_data))


# Create a new tab for SARIMA on Weekly Data with AutoARIMA option
if tabs == "SARIMA on Weekly Data":
    st.title("SARIMA on Weekly Data")

    # Input for SARIMA parameters
    p_value = st.number_input("Enter the p value (order of AR):", min_value=0)
    d_value = st.number_input("Enter the d value (order of differencing):", min_value=0)
    q_value = st.number_input("Enter the q value (order of MA):", min_value=0)
    P_value = st.number_input("Enter the P value (seasonal order of AR):", min_value=0)
    D_value = st.number_input("Enter the D value (seasonal order of differencing):", min_value=0)
    Q_value = st.number_input("Enter the Q value (seasonal order of MA):", min_value=0)
    seasonal_period = st.number_input("Enter the seasonal period (e.g., 7 for weekly data):", min_value=1)
    train_test_ratio = st.slider("Select Train-Test Split Ratio:", 0.1, 0.9, 0.7)

    # Option to run SARIMA manually
    if st.button("Run SARIMA Model Manually"):
        # Resample daily data to weekly data
        weekly_data = data['Close'].resample('W').mean().ffill()

        # Apply differencing to the data if not done already
        # if 'differenced_data' not in locals():
        #     differencing = st.number_input("Enter the differencing order:", min_value=0)
        #     differenced_data = weekly_data.diff(differencing).dropna()

        # Split data into train and test sets
        train_size = int(len(weekly_data) * train_test_ratio)
        train_data, test_data = weekly_data[:train_size], weekly_data[train_size:]

        # Fit SARIMA model to weekly data
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        sarima_model = SARIMAX(train_data, order=(p_value, d_value, q_value), seasonal_order=(P_value, D_value, Q_value, seasonal_period))
        sarima_model_fit = sarima_model.fit()

        # Make forecasts
        forecast = sarima_model_fit.forecast(steps=len(test_data))

        # Calculate evaluation metrics
        mae = mean_absolute_error(test_data, forecast)
        mse = mean_squared_error(test_data, forecast)
        rmse = math.sqrt(mse)
        rmse_values["SARIMA"] = rmse

        st.subheader("SARIMA Model Summary")
        st.write(sarima_model_fit.summary())

        st.subheader("SARIMA Model Evaluation on Weekly Data (Manual)")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Plot actual vs. forecasted values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_data.index, test_data.values, label="Actual")
        ax.plot(test_data.index, forecast, label="Forecast", color='orange')
        ax.legend()
        st.subheader("SARIMA Forecast vs. Actual on Weekly Data (Manual)")
        st.pyplot(fig)
          
if tabs == "LSTM":
            st.subheader(" Using LSTM Stock Forecasting")
            # List of key points
            st.markdown("### Key Points Why LSTM to be Used:")

            key_points = [
                "Sequential Data Handling:",
                "Long-Term Dependency:",
                "Noise and Volatility:",
                "Feature Learning:",
                "Recurrent Architecture:",
                "Adaptability:",
                "Irregular Time Gaps:"
            ]
            # Display the key points in columns
            col1, col2, col3 = st.columns(3)
            for i, point in enumerate(key_points[:7]):
                if i % 3 == 0:
                    col1.write(point)
                elif i % 3 == 1:
                    col2.write(point)
                else:
                    col3.write(point)

            # # Display the key points as bullet points
            
            # for point in key_points[:7]:
            #     st.markdown(f"- {point}")



                # Input for LSTM hyperparameters
            st.title("HyperParameters")
            look_back_info = (
              "Look-back Period:\n"
              "Definition: This hyperparameter determines how many previous time steps (or periods) "
              "the LSTM model should consider when making predictions.")
    
            num_epochs_info = (
              "Number of Epochs:\n"
              "Definition: An epoch is one complete pass through the entire training dataset "
              "during the training of a neural network.")
    
            batch_size_info = (
              "Batch Size:\n"
              "Definition: During training, the dataset is divided into smaller subsets called batches, "
              "and each batch is used to update the model's weights in one iteration.")

            st.markdown(look_back_info)
            st.markdown(num_epochs_info)
            st.markdown(batch_size_info)

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

            #   # Display the model summary
            # st.subheader("LSTM Model Summary")
            # st.text(model.summary())

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
            plt.title(f'{ticker} Stock Price Comparison (LSTM)')
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
            rmse_values["LSTM"] = rmse

            comparison_df = pd.DataFrame({
                 'Actual Stock Price': y_test.flatten(),
                 'LSTM Predicted Stock Price': predicted_stock_prices.flatten()},
                 index=test_dates)
            
            
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


# Conclusion Tab
if tabs == "Conclusion":
    st.subheader("Conclusion")

    # Provide a summary of your project and findings
    st.markdown("In this time series analysis project, we explored historical stock price data for a selected stock.")
    st.markdown("Here are the main findings and steps in the analysis:")
    st.markdown("1. **Data Visualization**")
    st.markdown("2. **Seasonal Decomposition**,**ACF and PACF Analysis**,**ADF Test**")
    st.markdown("3. **ARIMA Modeling**: We applied ARIMA modeling to make time series forecasts.")
    st.markdown("4. **AutoARIMA Modeling**: We used the pmdarima library to automatically select the best ARIMA model.")
    st.markdown("5. **SARIMA Modeling**: We also explored Seasonal ARIMA (SARIMA) modeling for weekly data.")
    st.markdown("6. **LSTM Modeling**: In addition to ARIMA, we applied Long Short-Term Memory (LSTM) neural network modeling for stock price forecasting.")
    st.markdown("7. **Evaluation**: We evaluated the accuracy of the models using metrics such as MAE, MSE, and RMSE.")
    st.markdown("8. **Conclusion**: We discussed the results and insights obtained from each modeling approach.")
    # Add other project steps and findings as needed

    # Include RMSE values for different models
    st.markdown("### Model RMSE Values")
    st.markdown("Lower RMSE values indicate better model performance.")

    st.markdown("**Note**:For more detailed information, please refer to the respective tabs for data visualization, modeling, and evaluation results.")

    # # Retrieve and display RMSE values from the dictionary
    # for model_name, rmse_value in rmse_values.items():
    #     if rmse_value is not None:
    #         st.markdown(f"- {model_name} RMSE: {rmse_value:.2f}")

    # Add any additional insights or findings here
    #st.markdown("Overall, the choice of the best forecasting model depends on the specific characteristics of the data and the forecasting horizon.")

    


# # Monthly Data Tab
# if tabs == "Monthly Data":
            
#             # Resample daily data to monthly frequency (taking the mean)
#             monthly_data = data['Adj Close'].resample('M').mean()
#             monthly_data = pd.DataFrame(monthly_data)
#             monthly_data

#             # Perform seasonal decomposition on monthly data
#             result_decom = seasonal_decompose(monthly_data, model="additive")

#             # Plot decomposed components
#             st.subheader("Decomposed Components")
#             fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
#             axes[0].plot(monthly_data.index, monthly_data, label='Original', color='blue')
#             axes[0].set_ylabel('Original')
#             axes[1].plot(monthly_data.index, result_decom.trend, label='Trend', color='green')
#             axes[1].set_ylabel('Trend')
#             axes[2].plot(monthly_data.index, result_decom.seasonal, label='Seasonal', color='red')
#             axes[2].set_ylabel('Seasonal')
#             axes[3].plot(monthly_data.index, result_decom.resid, label='Residual', color='purple')
#             axes[3].set_ylabel('Residual')
#             st.pyplot(fig)

#             # Interpret the results and provide insights
#             st.subheader("Insights")
#             if any(abs(result_decom.seasonal) > 0.1):  # You can adjust the threshold for seasonality detection
#                 st.write("The monthly data appears to have a seasonal component.")
#             else:
#                 st.write("No clear seasonal pattern detected in the monthly data.")


    
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


#Create a new tab for ARIMA

# if tabs == "Weekly":
#             #Resample daily data to monthly frequency (taking the mean)
#             monthly_data = data['Adj Close'].resample('W').mean()
#             monthly_data = pd.DataFrame(monthly_data)
#             monthly_data

#             period = st.number_input("Enter the seasonal decomposition period:", min_value=1)

#             differencing = st.number_input("Enter the differencing order:", min_value=0)


#             # Perform seasonal decomposition on monthly data
#             result_decom = seasonal_decompose(monthly_data, model="additive",period=period)

#             # Plot decomposed components
#             st.subheader("Decomposed Components")
#             fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
#             axes[0].plot(monthly_data.index, monthly_data, label='Original', color='blue')
#             axes[0].set_ylabel('Original')
#             axes[1].plot(monthly_data.index, result_decom.trend, label='Trend', color='green')
#             axes[1].set_ylabel('Trend')
#             axes[2].plot(monthly_data.index, result_decom.seasonal, label='Seasonal', color='red')
#             axes[2].set_ylabel('Seasonal')
#             axes[3].plot(monthly_data.index, result_decom.resid, label='Residual', color='purple')
#             axes[3].set_ylabel('Residual')
#             st.pyplot(fig)

#             # Interpret the results and provide insights
#             st.subheader("Insights")
#             if any(abs(result_decom.seasonal) > 0.1):  # You can adjust the threshold for seasonality detection
#                 st.write("The weekly data appears to have a seasonal component.")
#             else:
#                 st.write("No clear seasonal pattern detected in the weekly data.")

#             st.title("ACF and PACF Plots for Stock Price Data")

#             # Input for the number of lags
#             lags = 40
#              # Generate and display ACF and PACF plots
#             st.subheader("ACF and PACF Plots")
#             acf_plot, pacf_plot = acf_pacf_plot(monthly_data, lags)

#             st.pyplot(acf_plot)
#             st.pyplot(pacf_plot)

#               # Perform ADF test and display results
#             st.subheader("ADF Test Results")
#             adf_result = adfuller_test(monthly_data)
#             st.write(adf_result)

#             # differencing = 1
#             # Input for differencing
#             #differencing = st.number_input("Enter the differencing order:", min_value=0)
#             differenced_data = monthly_data.diff(differencing).dropna()
#             # Perform seasonal decomposition on differenced data
           
#             # trend, season, resid = decomposition(differenced_data,period=period)
#             # # Display the decomposed components of the differenced data
#             # st.subheader("Decomposed Components (Differenced Data)")
            
#             # st.write("Trend")
#             # st.line_chart(trend, use_container_width=True)
#             # st.write("Seasonal")
#             # st.line_chart(season, use_container_width=True)
#             # st.write("Residual")
#             # st.line_chart(resid, use_container_width=True)
            
#             # Plot the decomposed components with a custom legend
#             st.subheader("Decomposition Plot (Differenced Data)")
#             fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(10, 6))
            
#             result_decom = seasonal_decompose(differenced_data, model="additive",period=period)
            
#             result_decom.trend.plot(ax=axes[0],color="green")
#             axes[0].set_title("Trend")

#             result_decom.seasonal.plot(ax=axes[1],color="red")
#             axes[1].set_title("Seasonal")

#             result_decom.resid.plot(ax=axes[2],color="purple")
#             axes[2].set_title("Residual")

#             # Add a custom legend
#             for ax in axes:
#                 ax.legend(["Component"])
#             plt.tight_layout()
#             st.pyplot(fig)

#             # Generate and display ACF and PACF plots for differenced data
#             st.subheader("ACF and PACF Plots (Differenced Data)")
#             acf_plot, pacf_plot = acf_pacf_plot(differenced_data, lags)

#             st.pyplot(acf_plot)
#             st.pyplot(pacf_plot)

#             # Perform ADF test on differenced data and display results
#             st.subheader("ADF Test Results (Differenced Data)")
#             adf_result = adfuller_test(differenced_data)
#             st.write(adf_result)

#             st.write(len(monthly_data))

# # Create a new tab for SARIMA on Weekly Data
# if tabs == "SARIMA on Weekly Data":
#     st.title("SARIMA on Weekly Data")

#     # Input for SARIMA parameters
#     p_value = st.number_input("Enter the p value (order of AR):", min_value=0)
#     d_value = st.number_input("Enter the d value (order of differencing):", min_value=0)
#     q_value = st.number_input("Enter the q value (order of MA):", min_value=0)
#     P_value = st.number_input("Enter the P value (seasonal order of AR):", min_value=0)
#     D_value = st.number_input("Enter the D value (seasonal order of differencing):", min_value=0)
#     Q_value = st.number_input("Enter the Q value (seasonal order of MA):", min_value=0)
#     seasonal_period = st.number_input("Enter the seasonal period (e.g., 7 for weekly data):", min_value=1)
#     train_test_ratio = st.slider("Select Train-Test Split Ratio:", 0.1, 0.9, 0.7)

#     if st.button("Run SARIMA Model on Weekly Data"):
#         # Resample daily data to weekly data
#         weekly_data = data['Close'].resample('W').mean().ffill()

#         # Apply differencing to the data if not done already
#         # if 'differenced_data' not in locals():
#         #     differencing = st.number_input("Enter the differencing order:", min_value=0)
#         #     differenced_data = weekly_data.diff(differencing).dropna()

#         # Split data into train and test sets
#         train_size = int(len(weekly_data) * train_test_ratio)
#         train_data, test_data = weekly_data[:train_size], weekly_data[train_size:]

#         # Fit SARIMA model to weekly data
#         from statsmodels.tsa.statespace.sarimax import SARIMAX

#         model = SARIMAX(train_data, order=(p_value, d_value, q_value), seasonal_order=(P_value, D_value, Q_value, seasonal_period))
#         model_fit = model.fit()

#         # Make forecasts
#         forecast = model_fit.forecast(steps=len(test_data))
#         # forecast_mean = forecast.predicted_mean
#         # forecast_conf_int = forecast.conf_int()

#         # Calculate evaluation metrics
#         mae = mean_absolute_error(test_data, forecast)
#         mse = mean_squared_error(test_data, forecast)
#         rmse = math.sqrt(mse)

#         st.subheader("SARIMA Model Evaluation on Weekly Data")
#         st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
#         st.write(f"Mean Squared Error (MSE): {mse:.2f}")
#         st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

#         # Plot actual vs. forecasted values
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(test_data.index, test_data.values, label="Actual")
#         ax.plot(test_data.index, forecast, label="Forecast", color='orange')
#         #ax.fill_between(test_data.index, forecast_conf_int[:, 0], forecast_conf_int[:, 1], color='pink', alpha=0.5, label="Prediction Interval")
#         ax.legend()
#         st.subheader("SARIMA Forecast vs. Actual on Weekly Data")
#         st.pyplot(fig)

    # Option to run AutoSARIMA
    # if st.button("Run AutoSARIMA Model"):
    #     # Resample daily data to weekly data
    #     weekly_data = data['Close'].resample('W').mean().ffill()

    #     # Apply differencing to the data if not done already
    #     # if 'differenced_data' not in locals():
    #     #     differencing = st.number_input("Enter the differencing order:", min_value=0)
    #     #     differenced_data = weekly_data.diff(differencing).dropna()

    #     # Split data into train and test sets
    #     train_size = int(len(weekly_data) * train_test_ratio)
    #     train_data, test_data = weekly_data[:train_size], weekly_data[train_size:]

    #     # AutoSARIMA modeling and forecasting
    #     from pmdarima import auto_arima

    #     autosarima_model = auto_arima(train_data, seasonal=True, m=seasonal_period, trace=True, stepwise=True, suppress_warnings=True, error_action="ignore")
    #     autosarima_forecast = autosarima_model.predict(n_periods=len(test_data))

    #     # Calculate evaluation metrics for AutoSARIMA
    #     autosarima_mae = mean_absolute_error(test_data, autosarima_forecast)
    #     autosarima_mse = mean_squared_error(test_data, autosarima_forecast)
    #     autosarima_rmse = math.sqrt(autosarima_mse)

    #     st.subheader("AutoSARIMA Model Summary")
    #     st.write(autosarima_model.summary())

    #     st.subheader("AutoSARIMA Model Evaluation on Weekly Data")
    #     st.write(f"Mean Absolute Error (MAE): {autosarima_mae:.2f}")
    #     st.write(f"Mean Squared Error (MSE): {autosarima_mse:.2f}")
    #     st.write(f"Root Mean Squared Error (RMSE): {autosarima_rmse:.2f}")

    #     # Plot actual vs. forecasted values for AutoSARIMA
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     ax.plot(test_data.index, test_data.values, label="Actual")
    #     ax.plot(test_data.index, autosarima_forecast, label="AutoSARIMA Forecast", color='green')
    #     ax.legend()
    #     st.subheader("AutoSARIMA Forecast vs. Actual on Weekly Data")
    #     st.pyplot(fig)


# Create a single tab for ARIMA and SARIMA
# if tabs == "ARIMA & SARIMA":
#     st.title("ARIMA & SARIMA Analysis")

#     # Input for model selection (ARIMA or SARIMA)
#     model_type = st.radio("Select Model Type:", ["ARIMA", "SARIMA"])

#     # Input for model parameters
#     p_value = st.number_input("Enter the p value (order of AR):", min_value=0)
#     d_value = st.number_input("Enter the d value (order of differencing):", min_value=0)
#     q_value = st.number_input("Enter the q value (order of MA):", min_value=0)

#     if model_type == "SARIMA":
#         P_value = st.number_input("Enter the P value (seasonal order of AR):", min_value=0)
#         D_value = st.number_input("Enter the D value (seasonal order of differencing):", min_value=0)
#         Q_value = st.number_input("Enter the Q value (seasonal order of MA):", min_value=0)
#         seasonal_period = st.number_input("Enter the seasonal period (e.g., 12 for monthly data):", min_value=1)

#     train_test_ratio = st.slider("Select Train-Test Split Ratio:", 0.1, 0.9, 0.7)

#     if st.button(f"Run {model_type} Model"):
#         # Apply differencing to the data if not done already
#         # if 'differenced_data' not in locals():
#         #     differencing = st.number_input("Enter the differencing order:", min_value=0)
#         #     differenced_data = data['Close'].diff(differencing).dropna()

#             # Split data into train and test sets
#         train_size = int(len(data) * train_test_ratio)
#         train_data, test_data = data['Close'][:train_size], data['Close'][train_size:]

#         # Fit ARIMA or SARIMA model
#         if model_type == "ARIMA":
#             model = ARIMA(train_data, order=(p_value, d_value, q_value))
#         elif model_type == "SARIMA":
#             from statsmodels.tsa.statespace.sarimax import SARIMAX
#             model = SARIMAX(train_data, order=(p_value, d_value, q_value), seasonal_order=(P_value, D_value, Q_value,seasonal_period))
        
#         model_fit = model.fit()

#         # Make forecasts
#         if model_type == "ARIMA":
#             forecast = model_fit.forecast(steps=len(test_data))
#         elif model_type == "SARIMA":
#             forecast = model_fit.get_forecast(steps=len(test_data))
#         forecast_mean = forecast.predicted_mean
#         forecast_conf_int = forecast.conf_int()

#         # Calculate evaluation metrics
#         mae = mean_absolute_error(test_data, forecast_mean)
#         mse = mean_squared_error(test_data, forecast_mean)
#         rmse = math.sqrt(mse)

#         st.subheader(f"{model_type} Model Evaluation")
#         st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
#         st.write(f"Mean Squared Error (MSE): {mse:.2f}")
#         st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

#         # Plot actual vs. forecasted values
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(test_data.index, test_data.values, label="Actual")
#         ax.plot(test_data.index, forecast_mean, label="Forecast", color='orange')
#         ax.fill_between(test_data.index, forecast_conf_int[:, 0], forecast_conf_int[:, 1], color='pink', alpha=0.5, label="Prediction Interval")
#         ax.legend()
#         st.subheader(f"{model_type} Forecast vs. Actual")
#         st.pyplot(fig)



# if tabs == "AutoARIMA":
#             st.subheader("Auto_Arima")

#             # train_ratio = st.slider("Train Data Ratio (%)", 1, 99, 80)
#             # test_ratio = 100 - train_ratio

#             # # # Calculate the split index
#             # split_index = int(len(data) * train_ratio / 100)

#             # # # Split the data into train and test sets
#             # train_data = data.iloc[:split_index]
#             # test_data = data.iloc[split_index:]

#             # # Plot the data
#             # plt.figure(figsize=(10, 6))
#             # plt.grid(True)
#             # plt.xlabel('Date')
#             # plt.ylabel('Closing Price')
#             # plt.plot(train_data.index, train_data['Close'], 'green', label='Train data')
#             # plt.plot(test_data.index, test_data['Close'], 'blue', label='Test data')
#             # plt.title(f'{ticker} Train and Test Data')
#             # plt.legend()

#             # # Display the plot
#             # st.pyplot(plt)

#             # # Fit the autoARIMA model
#             # model_autoARIMA = auto_arima(train_data['Close'], start_p=0, start_q=0,
#             #                             test='adf', max_p=3, max_q=3,
#             #                             m=1, d=None, seasonal=False,
#             #                             start_P=0, D=0, trace=True,
#             #                             error_action='ignore',
#             #                             suppress_warnings=True,
#             #                             stepwise=True)

#             # # Display model summary
#             # st.subheader("AutoARIMA Model Summary")
#             # st.text(str(model_autoARIMA.summary()))

#             # # Plot model diagnostics
#             # st.subheader("AutoARIMA Model Diagnostics")
#             # model_autoARIMA.plot_diagnostics(figsize=(15, 8))
#             # st.pyplot(plt)

#             # # Fit an ARIMA model to the test data
#             # st.subheader("ARIMA Model and Forecast")
#             # p, d, q = model_autoARIMA.order
#             # model = ARIMA(train_data['Close'], order=(p, d, q))
#             # fitted = model.fit()

#             # # Display the model summary
#             # st.subheader("ARIMA Model Summary")
#             # st.text(str(fitted.summary()))

            
#             # # Forecast future values
#             # forecast_steps = len(test_data)  # Forecast the same number of steps as test data
#             # forecast = fitted.forecast(steps=forecast_steps, alpha=0.05)

#             # # Create forecast dates
#             # forecast_dates = pd.date_range(start=test_data.index[-1], periods=forecast_steps + 1)
#             # # Create a DataFrame for the forecasted data with the same date range as test_data
#             # forecast_df = pd.DataFrame({'Close': forecast}, index=forecast_dates)

#             # # Concatenate the test data and forecasted data
#             # combined_data = pd.concat([test_data, forecast_df])

#             # #Plot the forecast and actual test data
#             # plt.figure(figsize=(10, 6))
#             # plt.grid(True)
#             # plt.xlabel('Date')
#             # plt.ylabel('Closing Price')
#             # plt.plot(train_data.index, train_data['Close'], 'green', label='Train data')
#             # #plt.plot(test_data.index, test_data['Close'], 'blue', label='Test data')
#             # #plt.plot(forecast_dates[1:], forecast, 'orange', label='Forecasted data')
#             # #plt.plot(combined_data.index, combined_data['Close'], 'blue', label='Test data and Forecasted data')
#             # plt.plot(combined_data.index, combined_data['Close'], 'b-', label='Test data', linestyle='-')
#             # plt.plot(forecast_df.index, forecast_df['Close'], 'orange', label='Forecasted data')
#             # plt.title(f'{ticker} ARIMA Model Forecast')
#             # plt.legend()
#             # st.pyplot(plt)
#             #  # Create a DataFrame to combine the date range, actual test data, and forecasted values
#             # forecast_df = pd.DataFrame({
#             #     'Actual Stock Price': test_data['Close'].values,
#             #     'Forecasted Stock Price': forecast
#             # })

#             # # # Print the DataFrame
#             # st.subheader("Actual vs. Forecasted Stock Prices")
#             # st.write(forecast_df)

            
#             #     # Calculate and display performance metrics
#             # mse = mean_squared_error(test_data['Close'], forecast)
#             # mae = mean_absolute_error(test_data['Close'], forecast)
#             # rmse = math.sqrt(mse)
#             # #mape = np.mean(np.abs(forecast - test_data['Close']) / np.abs(test_data['Close']))
#             # #mape = np.mean(np.abs((test_data['Close'] - forecast) / test_data['Close'])) * 100
#             # # Calculate MAPE and accuracy
#             # mape = np.mean(np.abs((forecast_df['Actual Stock Price'] - forecast) / np.abs(forecast_df['Actual Stock Price']  + 1e-8))) * 100  
#             # # Added a small constant to avoid division by zero
#             # # Calculate accuracy score
#             # accuracy = 100 - mape

#             # st.subheader("Performance Metrics")
#             # st.write(f'MSE: {mse}')
#             # st.write(f'MAE: {mae}')
#             # st.write(f'RMSE: {rmse}')
#             # st.write(f'MAPE: {mape}')
#             # st.write(f'Accuracy: {accuracy:.2f}%')
#             # import pmdarima as pm


#             train_ratio = st.slider("Train Data Ratio (%)", 1, 99, 80)
#             test_ratio = 100 - train_ratio

#             # # Calculate the split index
#             split_index = int(len(data) * train_ratio / 100)

#             # # Split the data into train and test sets
#             train_data = data.iloc[:split_index]
#             test_data = data.iloc[split_index:]
#             st.title("Auto ARIMA Analysis")
#             # # Fit Auto ARIMA model
#             if st.button("Run Auto ARIMA Model"):
#                   model_autoARIMA = auto_arima(train_data['Close'], start_p=0, start_q=0,
#                                     test='adf', max_p=3, max_q=3,
#                                     m=1, d=None, seasonal=False,
#                                     start_P=0, D=0, trace=True,
#                                     error_action='ignore',
#                                     suppress_warnings=True,
#                                     stepwise=True)
#                    # Print the model summary
#                   st.subheader("Auto ARIMA Model Summary")
#                   st.text(model_autoARIMA.summary())

#                 # Make forecasts for the test data
#                   forecast, conf_int = model_autoARIMA.predict(n_periods=len(test_data), return_conf_int=True)
#                   # Convert date index to a list of strings for x-axis
#                   date_strings = [str(date) for date in test_data.index]
#                   test_data['Date'] = date_strings
#                       # Convert DataFrame columns to NumPy arrays
#                   test_data = test_data['Close'].to_numpy()
#                   forecast = forecast.ravel()
                  
#                 # Calculate evaluation metrics
#                   mae = mean_absolute_error(test_data, forecast)
#                   mse = mean_squared_error(test_data, forecast)
#                   rmse = math.sqrt(mse)
#                   st.subheader("Auto ARIMA Model Evaluation")
#                   st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
#                   st.write(f"Mean Squared Error (MSE): {mse:.2f}")
#                   st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

#                     # Create a DataFrame to display test data, forecasted data, and prediction intervals
#                   result_df = pd.DataFrame({'Date': date_strings,'Test Data': test_data, 'Forecasted Data': forecast})
#                   result_df['Lower Bound'] = conf_int[:, 0]
#                   result_df['Upper Bound'] = conf_int[:, 1]
#                   st.subheader("Test Data vs. Forecasted Data (with Prediction Intervals)")
#                   st.write(result_df)

#                     # Plot actual vs. forecasted values
#                   fig, ax = plt.subplots(figsize=(10, 6))
#                   #x_axis = range(len(test_data))
#                   ax.plot(date_strings, test_data, label="Actual")
#                   ax.plot(date_strings, forecast, label="Forecast", color='orange')
#                   ax.fill_between(date_strings, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.5, label="Prediction Interval")
#                   ax.legend()
#                   ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
#                   plt.xticks(rotation=45) 
#                   st.subheader("Auto ARIMA Forecast vs. Actual")
#                   st.pyplot(fig)

           
# if tabs == "ARIMA and Auto ARIMA":
#     st.title("ARIMA and Auto ARIMA Analysis")
#     st.subheader("Choose a Model")

#     # Input for ARIMA or Auto ARIMA
#     selected_model = st.radio("Select Model:", ("ARIMA", "Auto ARIMA"))
#     train_test_ratio = st.slider("Select Train-Test Split Ratio:", 0.1, 0.9, 0.7)

#      # Input for ARIMA parameters
#     p_value = st.number_input("Enter the p value (order of AR):", min_value=0)
#     d_value = st.number_input("Enter the d value (order of differencing):", min_value=0)
#     q_value = st.number_input("Enter the q value (order of MA):", min_value=0)

#     if st.button("Run Model"):
#         # Split data into train and test sets
#         train_size = int(len(data) * train_test_ratio)
#         train_data, test_data = data['Close'][:train_size], data['Close'][train_size:]
#         forecast = []  # Initialize forecast as an empty list
#         # date_strings = []
#         # forecast_dates = []

#         if selected_model == "ARIMA":
#             # # Input for ARIMA parameters
#             # p_value = st.number_input("Enter the p value (order of AR):", min_value=0)
#             # d_value = st.number_input("Enter the d value (order of differencing):", min_value=0)
#             # q_value = st.number_input("Enter the q value (order of MA):", min_value=0)

#             # Fit ARIMA model
#             model = ARIMA(train_data, order=(p_value, d_value, q_value))
#             model_fit = model.fit()

#             # Make forecasts
#             forecast= model_fit.forecast(steps=len(test_data))
#             #date_strings = [str(date) for date in test_data.index]
#             #forecast_dates = test_data.index

#             # Display ARIMA model summary
#             st.subheader("ARIMA Model Summary")
#             st.text(model_fit.summary())

#         elif selected_model == "Auto ARIMA":
#             # Fit Auto ARIMA model
#             from pmdarima import auto_arima

#             model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
#                                           test='adf', max_p=5, max_q=5,
#                                           m=1, d=None, seasonal=False,
#                                           start_P=0, D=0, trace=True,
#                                           error_action='ignore',
#                                           suppress_warnings=True,
#                                           stepwise=True)

#             # Print the model summary
#             st.subheader("Auto ARIMA Model Summary")
#             st.text(model_autoARIMA.summary())

#             # Make forecasts for the test data
#             forecast= model_autoARIMA.predict(n_periods=len(test_data))
#             #date_strings = [str(date) for date in test_data.index]
#             #forecast_dates = test_data.index

#         # Calculate evaluation metrics
#         mae = mean_absolute_error(test_data, forecast)
#         mse = mean_squared_error(test_data, forecast)
#         rmse = math.sqrt(mse)
#         rmse_values["ARIMA"] = rmse

#         fig, ax = plt.subplots(figsize=(10, 6))

#         # Plot training data
#         ax.plot(train_data.index, train_data.values, label="Train Data", color='green')
#         # Plot testing data
#         ax.plot(test_data.index, test_data.values, label="Test Data", color='blue')

#         # Create a DataFrame to display test data, forecasted data, and prediction intervals
#         #result_df = pd.DataFrame({'Date': date_strings,'Test Data': test_data, 'Forecasted Data': forecast})

#         # if selected_model == "Auto ARIMA":
#         #     # Add prediction intervals only for Auto ARIMA
#         #     result_df['Lower Bound'] = conf_int[:, 0]
#         #     result_df['Upper Bound'] = conf_int[:, 1]
      
  

#         #st.subheader("Test Data vs. Forecasted Data (with Prediction Intervals)")
#         #st.write(result_df)

#         # Plot actual vs. forecasted values
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(test_data.index,test_data.values, label="Actual")
#         ax.plot(test_data.index, forecast, label="Forecast", color='orange')
#         ax.legend()
#         st.subheader("Forecast vs. Actual")
#         st.pyplot(fig)

#         st.subheader("Model Evaluation")
#         st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
#         st.write(f"Mean Squared Error (MSE): {mse:.2f}")
#         st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")





# # Create a new tab for Auto ARIMA
# if tabs == "AutoARIMA":
#     st.title("Auto ARIMA Analysis")
            # #if st.button("Run Auto ARIMA Model"):
            #     # Fit Auto ARIMA model
            # arima_model = pm.auto_arima(train_data['Close'], 
            #                             seasonal=True,  # Enable seasonal decomposition
            #                             stepwise=True, 
            #                             trace=True,
            #                             suppress_warnings=True,
            #                             error_action="ignore",
            #                             max_order=None,  # Adjust as needed
            #                             random_state=42,  # Optional
            #                             n_jobs=-1)  # Optional
            # # Print the model summary
            # st.subheader("Auto ARIMA Model Summary")
            # st.text(arima_model.summary())
            # # Make forecasts for the test data
            # forecast, conf_int = arima_model.predict(n_periods=len(test_data), return_conf_int=True)
            #         # Convert date index to a list of strings for x-axis
            # date_strings = [str(date) for date in test_data.index]
            # test_data['Date'] = date_strings
            #         # Convert DataFrame columns to NumPy arrays
            # test_data = test_data['Close'].to_numpy()
            # forecast = forecast.ravel()

            # # Calculate evaluation metrics
            # mae = mean_absolute_error(test_data, forecast)
            # mse = mean_squared_error(test_data, forecast)
            # rmse = math.sqrt(mse)
            # st.subheader("Auto ARIMA Model Evaluation")
            # st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
            # st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            # st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

            # # Create a DataFrame to display test data, forecasted data, and prediction intervals
            # result_df = pd.DataFrame({'Date':date_strings,'Test Data': test_data, 'Forecasted Data': forecast})
            # result_df['Lower Bound'] = conf_int[:, 0]
            # result_df['Upper Bound'] = conf_int[:, 1]
            # st.subheader("Test Data vs. Forecasted Data (with Prediction Intervals)")
            # st.write(result_df)

            # # Plot actual vs. forecasted values
            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.plot(date_strings, test_data.values, label="Actual")
            # ax.plot(date_strings, forecast, label="Forecast", color='orange')
            # ax.fill_between(date_strings, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.5, label="Prediction Interval")
            # ax.legend()
            # st.subheader("Auto ARIMA Forecast vs. Actual")
            # st.pyplot(fig)




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





# # Create a new tab for SARIMA on Weekly Data
# if tabs == "SARIMA on Weekly Data":
#     st.title("SARIMA on Weekly Data")

#     # Input for SARIMA parameters
#     p_value = st.number_input("Enter the p value (order of AR):", min_value=0)
#     d_value = st.number_input("Enter the d value (order of differencing):", min_value=0)
#     q_value = st.number_input("Enter the q value (order of MA):", min_value=0)
#     P_value = st.number_input("Enter the P value (seasonal order of AR):", min_value=0)
#     D_value = st.number_input("Enter the D value (seasonal order of differencing):", min_value=0)
#     Q_value = st.number_input("Enter the Q value (seasonal order of MA):", min_value=0)
#     seasonal_period = st.number_input("Enter the seasonal period (e.g., 7 for weekly data):", min_value=1)
#     train_test_ratio = st.slider("Select Train-Test Split Ratio:", 0.1, 0.9, 0.7)

#     if st.button("Run SARIMA Model on Weekly Data"):
#         # Resample daily data to weekly data
#         weekly_data = data['Close'].resample('W').mean().ffill()

#         # Apply differencing to the data if not done already
#         # if 'differenced_data' not in locals():
#         #     differencing = st.number_input("Enter the differencing order:", min_value=0)
#         #     differenced_data = weekly_data.diff(differencing).dropna()

#         # Split data into train and test sets
#         train_size = int(len(weekly_data) * train_test_ratio)
#         train_data, test_data = weekly_data[:train_size], weekly_data[train_size:]

#         # Fit SARIMA model to weekly data
#         from statsmodels.tsa.statespace.sarimax import SARIMAX

#         model = SARIMAX(train_data, order=(p_value, d_value, q_value), seasonal_order=(P_value, D_value, Q_value, seasonal_period))
#         model_fit = model.fit()

#         # Make forecasts
#         forecast = model_fit.forecast(steps=len(test_data))
#         # forecast_mean = forecast.predicted_mean
#         # forecast_conf_int = forecast.conf_int()

#         # Calculate evaluation metrics
#         mae = mean_absolute_error(test_data, forecast)
#         mse = mean_squared_error(test_data, forecast)
#         rmse = math.sqrt(mse)

#         st.subheader("SARIMA Model Evaluation on Weekly Data")
#         st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
#         st.write(f"Mean Squared Error (MSE): {mse:.2f}")
#         st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

#         # Plot actual vs. forecasted values
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(test_data.index, test_data.values, label="Actual")
#         ax.plot(test_data.index, forecast, label="Forecast", color='orange')
#         #ax.fill_between(test_data.index, forecast_conf_int[:, 0], forecast_conf_int[:, 1], color='pink', alpha=0.5, label="Prediction Interval")
#         ax.legend()
#         st.subheader("SARIMA Forecast vs. Actual on Weekly Data")
#         st.pyplot(fig)

# # ... (previous code)

# # Create a new tab for SARIMA on Weekly Data with AutoARIMA option
# if tabs == "SARIMA on Weekly Data (Manual & Auto)":
#     st.title("SARIMA on Weekly Data (Manual & Auto)")

#     # Input for SARIMA parameters
#     p_value = st.number_input("Enter the p value (order of AR):", min_value=0)
#     d_value = st.number_input("Enter the d value (order of differencing):", min_value=0)
#     q_value = st.number_input("Enter the q value (order of MA):", min_value=0)
#     P_value = st.number_input("Enter the P value (seasonal order of AR):", min_value=0)
#     D_value = st.number_input("Enter the D value (seasonal order of differencing):", min_value=0)
#     Q_value = st.number_input("Enter the Q value (seasonal order of MA):", min_value=0)
#     seasonal_period = st.number_input("Enter the seasonal period (e.g., 7 for weekly data):", min_value=1)
#     train_test_ratio = st.slider("Select Train-Test Split Ratio:", 0.1, 0.9, 0.7)

#     # Option to run SARIMA manually
#     if st.button("Run SARIMA Model Manually"):
#         # Resample daily data to weekly data
#         weekly_data = data['Close'].resample('W').mean().ffill()

#         # Apply differencing to the data if not done already
#         # if 'differenced_data' not in locals():
#         #     differencing = st.number_input("Enter the differencing order:", min_value=0)
#         #     differenced_data = weekly_data.diff(differencing).dropna()

#         # Split data into train and test sets
#         train_size = int(len(weekly_data) * train_test_ratio)
#         train_data, test_data = weekly_data[:train_size], weekly_data[train_size:]

#         # Fit SARIMA model to weekly data
#         from statsmodels.tsa.statespace.sarimax import SARIMAX

#         sarima_model = SARIMAX(train_data, order=(p_value, d_value, q_value), seasonal_order=(P_value, D_value, Q_value, seasonal_period))
#         sarima_model_fit = sarima_model.fit()

#         # Make forecasts
#         forecast = sarima_model_fit.forecast(steps=len(test_data))

#         # Calculate evaluation metrics
#         mae = mean_absolute_error(test_data, forecast)
#         mse = mean_squared_error(test_data, forecast)
#         rmse = math.sqrt(mse)

#         st.subheader("SARIMA Model Summary")
#         st.write(sarima_model_fit.summary())

#         st.subheader("SARIMA Model Evaluation on Weekly Data (Manual)")
#         st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
#         st.write(f"Mean Squared Error (MSE): {mse:.2f}")
#         st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

#         # Plot actual vs. forecasted values
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(test_data.index, test_data.values, label="Actual")
#         ax.plot(test_data.index, forecast, label="Forecast", color='orange')
#         ax.legend()
#         st.subheader("SARIMA Forecast vs. Actual on Weekly Data (Manual)")
#         st.pyplot(fig)

#     # Option to run AutoSARIMA
#     if st.button("Run AutoSARIMA Model"):
#         # Resample daily data to weekly data
#         weekly_data = data['Close'].resample('W').mean().ffill()

#         # Apply differencing to the data if not done already
#         # if 'differenced_data' not in locals():
#         #     differencing = st.number_input("Enter the differencing order:", min_value=0)
#         #     differenced_data = weekly_data.diff(differencing).dropna()

#         # Split data into train and test sets
#         train_size = int(len(weekly_data) * train_test_ratio)
#         train_data, test_data = weekly_data[:train_size], weekly_data[train_size:]

#         # AutoSARIMA modeling and forecasting
#         from pmdarima import auto_arima

#         autosarima_model = auto_arima(train_data, seasonal=True, m=seasonal_period, trace=True, stepwise=True, suppress_warnings=True, error_action="ignore")
#         autosarima_forecast = autosarima_model.predict(n_periods=len(test_data))

#         # Calculate evaluation metrics for AutoSARIMA
#         autosarima_mae = mean_absolute_error(test_data, autosarima_forecast)
#         autosarima_mse = mean_squared_error(test_data, autosarima_forecast)
#         autosarima_rmse = math.sqrt(autosarima_mse)

#         st.subheader("AutoSARIMA Model Summary")
#         st.write(autosarima_model.summary())

#         st.subheader("AutoSARIMA Model Evaluation on Weekly Data")
#         st.write(f"Mean Absolute Error (MAE): {autosarima_mae:.2f}")
#         st.write(f"Mean Squared Error (MSE): {autosarima_mse:.2f}")
#         st.write(f"Root Mean Squared Error (RMSE): {autosarima_rmse:.2f}")

#         # Plot actual vs. forecasted values for AutoSARIMA
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(test_data.index, test_data.values, label="Actual")
#         ax.plot(test_data.index, autosarima_forecast, label="AutoSARIMA Forecast", color='green')
#         ax.legend()
#         st.subheader("AutoSARIMA Forecast vs. Actual on Weekly Data")
#         st.pyplot(fig)



# if tabs == "Monthly Arima":
#             monthly_data = data['Adj Close'].resample('M').mean()
#             monthly_data = pd.DataFrame(monthly_data)
#             monthly_data
#             st.subheader("Auto_Arima")


#             train_ratio = st.slider("Train Data Ratio (%)", 1, 99, 80)
#             test_ratio = 100 - train_ratio

#             # Calculate the split index
#             split_index = int(len(monthly_data) * train_ratio / 100)

#             # Split the data into train and test sets
#             train_data = monthly_data.iloc[:split_index]
#             test_data = monthly_data.iloc[split_index:]

#             # Plot the data
#             plt.figure(figsize=(10, 6))
#             plt.grid(True)
#             plt.xlabel('Date')
#             plt.ylabel('Closing Price')
#             plt.plot(train_data.index, train_data['Adj Close'], 'green', label='Train data')
#             plt.plot(test_data.index, test_data['Adj Close'], 'blue', label='Test data')
#             plt.title(f'{ticker} Train and Test Data')
#             plt.legend()

#             # Display the plot
#             st.pyplot(plt)

#             # Fit the autoARIMA model
#             model_autoARIMA = auto_arima(train_data['Adj Close'], start_p=0, start_q=0,
#                                         test='adf', max_p=3, max_q=3,
#                                         m=1, d=None, seasonal=False,
#                                         start_P=0, D=0, trace=True,
#                                         error_action='ignore',
#                                         suppress_warnings=True,
#                                         stepwise=True)

#             # Display model summary
#             st.subheader("AutoARIMA Model Summary")
#             st.text(str(model_autoARIMA.summary()))

#             # Plot model diagnostics
#             st.subheader("AutoARIMA Model Diagnostics")
#             model_autoARIMA.plot_diagnostics(figsize=(15, 8))
#             st.pyplot(plt)

#             # Fit an ARIMA model to the test data
#             st.subheader("ARIMA Model and Forecast")
#             p, d, q = model_autoARIMA.order
#             model = ARIMA(train_data['Adj Close'], order=(p, d, q))
#             fitted = model.fit()

#             # Display the model summary
#             st.subheader("ARIMA Model Summary")
#             st.text(str(fitted.summary()))

            
#             # Forecast future values
#             forecast_steps = len(test_data)  # Forecast the same number of steps as test data
#             forecast1 = fitted.forecast(steps=forecast_steps, alpha=0.05)

#             # Create forecast dates
#             forecast_dates = pd.date_range(start=test_data.index[-1], periods=forecast_steps + 1)
#             # Create a DataFrame for the forecasted data with the same date range as test_data
#             forecast_df = pd.DataFrame({'Adj Close': forecast1}, index=forecast_dates)

#             # Concatenate the test data and forecasted data
#             combined_data = pd.concat([test_data, forecast_df])

#             # #Plot the forecast and actual test data
#             # plt.figure(figsize=(10, 6))
#             # plt.grid(True)
#             # plt.xlabel('Date')
#             # plt.ylabel('Closing Price')
#             # plt.plot(train_data.index, train_data['Adj Close'], 'green', label='Train data')
#             # #plt.plot(test_data.index, test_data['Close'], 'blue', label='Test data')
#             # #plt.plot(forecast_dates[1:], forecast, 'orange', label='Forecasted data')
#             # #plt.plot(combined_data.index, combined_data['Close'], 'blue', label='Test data and Forecasted data')
#             # plt.plot(combined_data.index, combined_data['Adj Close'], 'b-', label='Test data', linestyle='-')
#             # plt.plot(forecast_df.index, forecast_df['Adj Close'], 'orange', label='Forecasted data')
#             # plt.title(f'{ticker} ARIMA Model Forecast')
#             # plt.legend()
#             # st.pyplot(plt)

#             # Plot the forecast and actual test data
#             plt.figure(figsize=(10, 6))
#             plt.grid(True)
#             plt.xlabel('Date')
#             plt.ylabel('Closing Price')
#             plt.plot(train_data.index, train_data['Adj Close'], 'green', label='Train data')
#             plt.plot(test_data.index, test_data['Adj Close'], 'blue', label='Test data')
#             plt.plot(forecast_dates[1:], forecast1, 'orange', label='Forecasted data')  # Plot the forecasted data
#             plt.title(f'{ticker} ARIMA Model Forecast')
#             plt.legend()
#             st.pyplot(plt)

#              # Create a DataFrame to combine the date range, actual test data, and forecasted values
#             forecast_df = pd.DataFrame({
#                 'Actual Stock Price': test_data['Adj Close'].values,
#                 'Forecasted Stock Price': forecast1
#             })

#             # # Print the DataFrame
#             st.subheader("Actual vs. Forecasted Stock Prices")
#             st.write(forecast_df)

#             # #Forecast future values
#             #forecast_steps = st.write("number of forecast steps:", len(test_data))
#             # #forecast_steps = st.number_input("Enter the number of forecast steps:", min_value=1)
#             # forecast = fitted.forecast(steps=forecast_steps, alpha=0.05)
#             #forecast_dates = pd.date_range(start=test_data.index[-1], periods=forecast_steps)
#             # forecast_dates = pd.date_range(start=test_data.index[-1], periods=forecast_steps + 1, freq='D')[1:]

#             # if len(forecast) > len(test_data):
#             #         forecast = forecast[:len(test_data)]

#             #     # Plot the forecast and actual test data
#             # plt.figure(figsize=(10, 5), dpi=100)
#             # plt.plot(test_data.index, test_data['Close'], label='Actual Stock Price', color='blue')
#             # plt.plot(forecast_dates, forecast, label='Predicted Stock Price', color='orange')
#             # plt.title(f'{ticker} Stock Price Comparison')
#             # plt.xlabel('Time')
#             # plt.ylabel('Stock Price')
#             # plt.legend()
#             # st.pyplot(plt)

#             # #Plot the forecast
#             # plt.figure(figsize=(10, 5), dpi=100)
#             # plt.plot(train_data.index, train_data['Close'], label='Training data')
#             # plt.plot(test_data.index, test_data['Close'], color='blue', label='Actual Stock Price')
#             # plt.plot(forecast_dates, forecast, color='orange', label='Predicted Stock Price')
#             # #plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=0.1)
#             # plt.title(f'{ticker} Stock Price Prediction')
#             # plt.xlabel('Time')
#             # plt.ylabel('Stock Price')
#             # plt.legend()
#             # st.pyplot(plt)

#                 # Calculate and display performance metrics
#             mse = mean_squared_error(test_data['Adj Close'], forecast1)
#             mae = mean_absolute_error(test_data['Adj Close'], forecast1)
#             rmse = math.sqrt(mse)
#             #mape = np.mean(np.abs(forecast - test_data['Close']) / np.abs(test_data['Close']))
#             mape = np.mean(np.abs((test_data['Adj Close'] - forecast1) / test_data['Adj Close'])) * 100
#             # Calculate accuracy score
#             accuracy = 100 - mape

#             st.subheader("Performance Metrics")
#             st.write(f'MSE: {mse}')
#             st.write(f'MAE: {mae}')
#             st.write(f'RMSE: {rmse}')
#             st.write(f'MAPE: {mape}')
#             st.write(f'Accuracy: {accuracy:.2f}%')


# if tabs == "Monthly Analysis":
#     st.title("Monthly Analysis")

#     # # Convert the date column to datetime format and set it as the index
#     # data['Date'] = pd.to_datetime(data['Date'])
#     # data.set_index('Date', inplace=True)

#     # Resample data to monthly frequency
#     data_monthly = data['Close'].resample('M').mean()

#     # Display the first few rows of the monthly data
#     st.subheader("Monthly Data")
#     st.write(data_monthly.head())

#     # Seasonal decomposition of monthly data
#     decomposition = seasonal_decompose(data_monthly, model='additive', period=12)  # Assumes yearly seasonality

#     # Plot seasonal, trend, and residual components
#     st.write("Trend Component")
#     st.line_chart(decomposition.trend)

#     st.write("Seasonal Component")
#     st.line_chart(decomposition.seasonal)

#     st.write("Residual Component")
#     st.line_chart(decomposition.resid)

#     # Check for seasonality in the monthly data
#     st.subheader("Check for Seasonality")
#     st.write("Plot of Monthly Data")
#     plt.figure(figsize=(10, 6))
#     plt.plot(data_monthly.index, data_monthly.values)
#     plt.xlabel("Date")
#     plt.ylabel("Close Price")
#     plt.title("Monthly Data Plot")
#     st.pyplot(plt)

#     # ACF and PACF plots
#     st.write("ACF (Autocorrelation Function) Plot")
#     fig_acf, ax_acf = plt.subplots(figsize=(10, 6))
#     plot_acf(data_monthly, lags=40,ax = ax_acf)
#     st.pyplot(fig_acf)

#     st.write("PACF (Partial Autocorrelation Function) Plot")
#     fig_pacf, ax_pacf = plt.subplots(figsize=(10, 6))
#     plot_pacf(data_monthly, lags=40,ax = ax_pacf)
#     st.pyplot(fig_pacf)

#     # Augmented Dickey-Fuller Test for stationarity
#     st.subheader("Augmented Dickey-Fuller Test for Stationarity")
#     result = adfuller_test(data_monthly)
#     st.write(result)
#     # result = adfuller(data_monthly, autolag="AIC")
#     # st.write("ADF Statistic:", result[0])
#     # st.write("p-value:", result[1])
#     # st.write("Critical Values:")
#     # for key, value in result[4].items():
#     #     st.write(f"{key}: {value}")

#     # Check for stationarity and apply differencing if needed
#     if result[1] > 0.05:
#         st.write("Data is not stationary. Applying differencing...")
#         differencing_order = st.number_input("Enter the differencing order:", min_value=0)
#         data_monthly_diff = data_monthly.diff(differencing_order).dropna()

#         # ACF and PACF plots for differenced data
#         st.write("ACF (Autocorrelation Function) Plot (Differenced Data)")
#         fig_acf_diff, ax_acf_diff = plt.subplots(figsize=(10, 6))
#         plot_acf(data_monthly_diff, lags=40, ax = ax_acf_diff)
#         st.pyplot(fig_acf_diff)

#         st.write("PACF (Partial Autocorrelation Function) Plot (Differenced Data)")
#         fig_pacf_diff, ax_pacf_diff = plt.subplots(figsize=(10, 6))
#         plot_pacf(data_monthly_diff, lags=40,ax = ax_pacf_diff)
#         st.pyplot(fig_pacf_diff)

#         # Augmented Dickey-Fuller Test for stationarity on differenced data
#         st.subheader("Augmented Dickey-Fuller Test for Stationarity (Differenced Data)")
#         result = adfuller_test(data_monthly_diff)
#         st.write(result)
#     else:
#         st.write("Data is stationary. No differencing needed.")

#     # Additional analysis and visualizations can be added here for the monthly data





           



          

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





