# import pandas libraries
import pandas as pd
# split data
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
# create visualizations
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)
# load the energy_usage.csv into pandas data set
data_set = pd.read_csv('energy_usage.csv')

# print first 5 rows of data set
print(data_set.head())

# inputs , (what predics energy consumption)
x = data_set[['hour_of_day', 'day_of_week']]
# outputs, pretty much what we want to predict
y = data_set['energy_consumed(hour)']

# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# create a linear regression model used to model the relationships between the input and output
model = LinearRegression()

# Train the model with the training data
model.fit(X_train, y_train)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')
# Route for handling the form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    hour = int(request.form['hour'])
    day_of_week = int(request.form['day_of_week'])
    
    # Use the trained model to make a prediction
    prediction = model.predict([[hour, day_of_week]])[0]
    
    # Recommendation based on prediction
    if prediction > 1.0:
        recommendation = "Try using this appliance earlier or later to save energy."
    else:
        recommendation = "Energy usage is already low. Go ahead!"
    
    # Return the result to the user
    return render_template('index.html', prediction=prediction, recommendation=recommendation)

if __name__ == "__main__":
    app.run(debug=True)



