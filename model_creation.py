import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from scipy.stats import linregress
import datetime

    
def get_data(filename):
    # Read in the Historical Data for the S&P 500
    sp500 = pd.read_csv(filename, usecols=['Date','Open','Day High','Day Low','Close', 'Volume'])
    
    # Format the column data types
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    
    # Add an id column
    index_values = sp500.index
    sp500.insert(0, 'Id', index_values)
    
    # Add a column that calculates the percent increase from opening to closing
    sp500['Day_Difference'] = ((sp500['Close'] - sp500['Open']) / sp500['Open']) * 100
    
    # add columns that represent the year, quarter, month, and day
    sp500['Year'] = sp500['Date'].dt.year
    sp500['Quarter'] = sp500['Date'].dt.quarter
    sp500['Month'] = sp500['Date'].dt.month
    sp500['Day'] = sp500['Date'].dt.day
    
    # Add a column that uses the slope of the preceding closing values to predict the next closing
    for i in range(len(sp500)):
        # Use only the data leading up to the point
        leading_data = sp500.loc[:i]
        # Find the slope and intercept 
        slope, intercept, _, _, _ = linregress(leading_data['Id'], leading_data['Open'])
        # Get the predicted value
        predicted_value = slope * (i + 1) + intercept
        
        sp500.at[0, 'Slope_Prediction'] = sp500.at[0, 'Open']
        sp500.at[i, 'Slope_Prediction'] = predicted_value
    
    # Add a column that shows how much higher or lower the stock price is than the slope predicted price 
    sp500['Predicted_Difference'] = ((sp500['Open'] - sp500['Slope_Prediction']) / sp500['Open']) * 100
    
    # Split the data into training and testing sets
    training_data = sp500[sp500['Date'] < datetime.datetime(2017, 1, 2)]
    testing_data = sp500[sp500['Date'] >= datetime.datetime(2017, 1, 2)]
    
    return sp500, training_data, testing_data
    
## Making Graphs to visualize the data

def visualize_data(sp500):
    
    plt.plot(sp500['Date'], sp500['Open'])
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    
    plt.plot(sp500['Date'], sp500['Day_Difference'])
    plt.xlabel('Date')
    plt.ylabel('Difference by Day as a Percentage')
    plt.show()
    
    plt.plot(sp500['Date'], sp500['Slope_Prediction'])
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.show()
    
    plt.plot(sp500['Date'], sp500['Predicted_Difference'])
    plt.xlabel('Date')
    plt.ylabel('Predicted difference as a percentage')
    plt.axhline(0, color='black', linestyle='dotted')
    plt.show()


def train_model(training_data):
    reg = LinearRegression()

    # Training the model
    labels = training_data['Day_Difference']
    train_data = training_data.drop(['Day Low', 'Date', 'Day High', 'Volume', 'Close', 'Day_Difference'], axis=1)

    reg.fit(train_data, labels)

    model = ensemble.GradientBoostingRegressor(
        n_estimators=3000,
        max_features="log2",
        max_depth=6,
        min_samples_leaf=9,
        learning_rate=0.1,
        loss='huber'
    )

    model.fit(train_data, labels)

    return model

    
def test_model(model, test_data):
    # Separate the features and target variable for the test data
    test_features = test_data.drop(['Day Low', 'Date', 'Day High', 'Volume', 'Close', 'Day_Difference'], axis=1)
    test_target = test_data['Day_Difference']

    # Make predictions using the trained model
    y_pred = model.predict(test_features)

    # Calculate the accuracy of the model on the test data
    accuracy = model.score(test_features, test_target)

    # Create a copy of the test_data DataFrame with predicted values
    predicted_data = test_data.copy()
    predicted_data['Predicted_Difference'] = y_pred

    # Plot the actual and predicted values
    start_date = datetime.datetime(2017, 1, 2)
    end_date = datetime.datetime(2023, 5, 24)
    
    plt.plot(predicted_data['Date'], predicted_data['Day_Difference'], label='Actual')
    plt.plot(predicted_data['Date'], predicted_data['Predicted_Difference'], label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Difference by Day as a Percentage')
    plt.xlim(start_date, end_date)
    plt.legend()
    plt.show()

    print("Accuracy:", accuracy)


if __name__ == '__main__':
    sp500, training_data, testing_data = get_data('HistoricalData.csv')
    model = train_model(training_data)
    test_model(model, testing_data)
