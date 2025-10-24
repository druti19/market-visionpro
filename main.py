# Import necessary libraries
import hashlib  # Library for hashing passwords
import numpy as np
import pandas as pd
import os
from flask import Flask, flash, request, redirect, url_for, render_template  # Flask for web development
import yfinance as yf  # Library for fetching stock data

# Set a seed for reproducibility
np.random.seed(42)

# Set the output directory for storing stock data
output_directory = r'C:\\Users\kbrek\\PycharmProjects\\search_predictionsFTSE100\\stock_data_for_FTSE500'
os.makedirs(output_directory, exist_ok=True)

# Initialize Flask app
app = Flask(__name__, template_folder='C:\\Users\\kbrek\\PycharmProjects\\my_LSTM\\Templates')

# Configure file upload settings
UPLOAD_FOLDER = 'C:\\Users\\kbrek\\PycharmProjects\\my_LSTM\\file_upload'
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed file extensions
ALLOWED_EXTENSIONS = set(['csv', 'xls', 'xlsx'])

# Define the path for storing user registration data
USER_FILE = 'C:\\Users\\kbrek\\PycharmProjects\\my_LSTM\\users_data\\users_data.csv'


# Define a class for user registration form
class RegistrationForm:
    def __init__(self, username, password, confirm):
        self.username = username
        self.password = password
        self.confirm = confirm


# Function to hash a password using MD5
def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()


# Function to verify a password against its hashed version
def verify_password(hashed_password, password):
    return hashed_password == hashlib.md5(password.encode()).hexdigest()


# Function to read user data from a file
def read_user_data():
    try:
        with open(USER_FILE, 'r') as file:
            lines = file.readlines()
            # Create a dictionary from user data file
            return {line.split(',')[0]: line.split(',')[1].strip() for line in lines}
    except FileNotFoundError:
        return {}
    except Exception as e:
        # Handle exceptions and display an error message
        flash(f"Error reading user data: {e}", 'error')
        return {}


# Function to write user data to a file
def write_user_data(user_data):
    try:
        with open(USER_FILE, 'w') as file:
            file.write('Username,Hashed_Password\n')
            # Write username and hashed password to the file
            for username, hashed_password in user_data.items():
                file.write(f'{username},{hashed_password}\n')
    except Exception as e:
        # Handle exceptions and display an error message
        flash(f"Error writing user data: {e}", 'error')


# Function to validate a registration form
def validate_registration_form(username, password, confirm):
    errors = []  # List to store validation errors

    # Check username length
    if len(username) < 4 or len(username) > 25:
        errors.append('Username must be between 4 and 25 characters.')

    # Check if password is empty
    if not password:
        errors.append('Password is required.')

    # Check if passwords match
    if password != confirm:
        errors.append('Passwords must match.')

    return errors


# Define the route for the home page
@app.route('/')
def home():
    # Set background image URL for the home page
    background_image_url = url_for('static', filename='vecteezy_abstract-gradient-pastel-blue-and-purple-background'
                                                      '-neon_8617161-1.jpg')
    return render_template('home.html', background_image_url=background_image_url)


# Define the route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Set background image URL for the registration page
    background_image_url = url_for('static', filename='vecteezy_abstract-blue-background-simple-design-for-your-website_6852804.jpg')

    if request.method == 'POST':
        # Get user registration form data
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm']

        # Read existing user data
        user_data = read_user_data()

        if username in user_data:
            # Display an error message if the username already exists
            error_message = "Username already exists. Please choose a different username."
            flash(error_message, 'error')
        else:
            # Validate the registration form
            validation_errors = validate_registration_form(username, password, confirm)

            if validation_errors:
                # Display validation errors
                for error in validation_errors:
                    flash(error, 'error')
            else:
                # Hash the password and store user data
                password_hash = hash_password(password)
                user_data[username] = password_hash
                write_user_data(user_data)
                flash("Registration successful. You can now login.", 'success')
                return redirect(url_for('login'))

    return render_template('register.html', background_image_url=background_image_url)


# Define the route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get login form data
        username = request.form['username']
        password = request.form['password']

        # Read user data for login validation
        user_data = read_user_data()

        if username in user_data and verify_password(user_data[username], password):
            # Redirect to home page upon successful login
            return redirect(url_for('home2'))
        else:
            # Display login failure message
            error_message = "Login failed. Please check your username and password."
            flash(error_message)
            # Set background image URL for the login page
            background_image_url = url_for('static',
                                           filename='vecteezy_abstract-blue-background-simple-design-for-your-website_6852804.jpg')
            return render_template('login.html', error_message=error_message, background_image_url=background_image_url)

    background_image_url = url_for('static',
                                   filename='vecteezy_abstract-blue-background-simple-design-for-your-website_6852804.jpg')
    return render_template('login.html', error_message="",
                           background_image_url=background_image_url)



def data_prep(df, duration):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    last_record = df.tail(1)
    last_record_date = pd.to_datetime(last_record["Date"].iloc[0])
    one_year_ago = last_record_date - pd.DateOffset(months=18)
    df_one_year_ago = df[df['Date'] >= one_year_ago]
    print("length of one_year_ago: ", len(df_one_year_ago))
    df_one_year_ago.index = df_one_year_ago['Date']

    date_train = df_one_year_ago['Date']

    # exception handling
    try:
        df_train = df_one_year_ago[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    except KeyError as e:
        df_train = df_one_year_ago[['Date', 'Open']]

    print("Dataset trained: ", df_train)
    open_prices = df_train.loc[:, 'Open'].values

    df_train = open_prices.astype(float)  #instance of a 2Darray

    df_train_normalized, mean, std = normalize_inputs(df_train)

    X, y = [], []

    print("df_train normalised values: ", df_train_normalized)

    for i in range(1, duration):
        X.append(df_train_normalized[i - 1])   #instance of a 2D array
        y.append(df_train_normalized[i:i + 1])   #instance of a 2D array

    # use of arrays in x and Y
    return np.array(X), np.array(y), df_train, date_train, mean, std


def normalize_inputs(data):
    # function that normalises the data
    total = sum(data)
    N = len(data)
    mean = total / N
    squared_diff_sum = sum((data - mean) ** 2)  # where 2 = beta
    N = len(data)
    std = (squared_diff_sum / N) ** 0.5  # where 0.5= alpha
    z = (data - mean) / std  # formula used is: Z = (data-mean) / std
    return z, mean, std


# use of a neural network
class LSTM:
    def __init__(self, input_columns, input_size, hidden_size, output_size, num_epochs, learning_rate):
        # Class initialization of LSTM
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs

        self.initialise_weights(input_size, hidden_size, input_columns, output_size)

    def initialise_weights(self, input_size, hidden_size, input_columns, output_size):
        # Is a function used for initializing weights in the LSTM Multi-dimensional arrays used for initialising each
        # of the weights and biases
        # Achieving the Objective 1.1.1: Initialise weights and biases for the LSTM model
        # and set up the parameters required for the forward propagation, including forget, input, candidate,
        # and output gates.
        self.weights_forget = self.initialiseWeights(input_size, hidden_size + input_columns)
        self.bias_forget = np.random.randn(hidden_size, 1) * 0.01

        self.weights_input = self.initialiseWeights(input_size, hidden_size + input_columns)
        self.bias_input = np.random.randn(hidden_size, 1) * 0.01

        self.weights_candidate = self.initialiseWeights(input_size, hidden_size + input_columns)
        self.bias_candidate = np.random.randn(hidden_size, 1) * 0.01

        self.weights_output = self.initialiseWeights(input_size, hidden_size + input_columns)
        self.bias_output = np.random.randn(hidden_size, 1) * 0.01

        self.weight_final = self.initialiseWeights(hidden_size, output_size)
        self.bias_final = np.random.randn(output_size, 1) * 0.01

    # Achieving Objective 1.2.1: Implement input data preprocessing to feed sequential information to the LSTM model through using the activation
    # functions such as tanh, sigmoid, and intialising weights
    def initialiseWeights(self,input_size, output_size):
        # activation function to initalise weights
        # 1/ sqrt(output_size + input_size)
        return np.random.randn(output_size, input_size) / np.sqrt(input_size + output_size)

    def sigmoid(self, input, derivative=False):
        # a type of activation function
        if derivative:
            return input * (1 - input)

        # σ(x)= 1/1+e^−x
        return 1 / (1 + np.exp(-input))

    def tanh(self ,input, derivative=False):
        # another type of activation function
        if derivative:
            return 1 - input ** 2

        return np.tanh(input)

    def variables(self):
        # Lists: Sequentially storing hidden, cell, candidate, outputgate, forget, input_gates, and outputs
        # Achieving the Objective 1.1.1: Initialize weights and biases for the LSTM model and set up the parameters
        # required for the forward propagation, including forget, input, candidate, and output gates.
        self.concatenated_inputs = {}

        self.hidden = {-1: np.zeros((self.hidden_size, 1))}
        self.cell = {-1: np.zeros((self.hidden_size, 1))}
        self.candidate = {}
        self.outputgate = {}
        self.forget = {}
        self.input_gates = {}
        self.outputs = {}

    def forward_propogation(self, inputs):
        # A function is used here for forward propagation in the LSTM
        # Achieiving Objective 4.1.1: Write the code for the forward propagation and define the architecture
        # with appropriate layers, neurons, and activation functions using the formulas for the forget gate,
        # input gate, candidate gate, output gate, updated cell state, and hidden state.
        self.variables()

        # Loop created iterating over the length of inputs
        outputs = []
        for q in range(len(inputs)):
            # An array is used here for storing the values for concatenated_inputs, forget, input_gates, candidate,
            # outputgate, cell, hidden, and outputs the array is being added to as each loop is continued
            self.concatenated_inputs[q] = np.concatenate((self.hidden[q - 1], inputs[q].reshape(-1, 1))).reshape(-1, 1)
            self.forget[q] = self.computing_gradients_using_sigmoid(self.weights_forget.T, self.concatenated_inputs[q],
                                                                    self.bias_forget)
            # comes from the formula:f of t = σ((W of if ⋅ x of t) + b of if + (W of hf ⋅ h of t−1) + b of hf)
            self.input_gates[q] = self.computing_gradients_using_sigmoid(self.weights_input.T,
                                                                         self.concatenated_inputs[q]
                                                                         , self.bias_input)
            # comes from the formula: i of t = σ((W of ii ⋅ x of t) + b of ii +(W of hi ⋅ h of t-1) + b of hi)
            self.candidate[q] = self.tanh(np.dot(self.weights_candidate.T, self.concatenated_inputs[q]) + self.bias_candidate)
            # comes from the formula: C~t = tanh((W of ig ⋅ x of t) + b of ig + (W of hg ⋅ h of t-1) + b of hg)
            self.outputgate[q] = self.computing_gradients_using_sigmoid(self.weights_output.T,
                                                                        self.concatenated_inputs[q],
                                                                        self.bias_output)
            # comes from the formula: o of t = σ((W of io ⋅ x of t) + b of io + (W of ho ⋅ h of t-1 + b of ho))
            self.cell[q] = self.forget[q] * self.cell[q - 1] + self.input_gates[q] * self.candidate[q]
            # comes from the formula: C of t = (f of t ⋅ C of t-1) + (i of t ⋅ C~t)
            self.hidden[q] = self.outputgate[q] * self.tanh(self.cell[q])

            # comes from the formula: h of t = (o of t ⋅ tanh(C of t))
            outputs += [np.dot(self.weight_final, self.hidden[q]) + self.bias_final]

        return outputs

    def computing_gradients_using_sigmoid(self, weights, concatenated_inputs, bias):
        # A function is used for computing the gradients using the sigmoid function
        y = self.sigmoid(np.dot(weights, concatenated_inputs) + bias)

        return y

    def backward_propogation(self, errors, inputs):
        # Function used for backward propagation in the LSTM
        # Arrays are initialised here for all weights and bias derivatives
        b_wforget, b_bforget = 0, 0
        b_winput, b_binput = 0, 0
        b_wcandidate, b_bcandidate = 0, 0
        b_woutput, b_boutput = 0, 0
        b_wfinal, b_bfinal = 0, 0

        b_updatehidden, b_updatecandidate = np.zeros_like(self.hidden[0]), np.zeros_like(self.cell[0])
        for q in reversed(range(len(inputs))):
            error = errors[q]

            b_wfinal += np.dot(error, self.hidden[q].T)
            b_bfinal += error

            b_hiddenState = np.dot(self.weight_final.T, error) + b_updatehidden

            b_outputState = self.calculate_new_state(self.tanh(self.cell[q]), b_hiddenState, self.sigmoid(self.outputgate[q],
                                                                                               derivative=True))
            # comes from the formula: ∂E/∂ot = ∂E / ∂(h of t) ⋅ tanh(C of t) ⋅ σ′(o of t)
            reshaped_input_q = inputs[q].reshape(-1, 1)
            b_woutput += np.dot(b_outputState, reshaped_input_q.T)
            b_boutput += b_outputState

            # Achieivng Objective 4.2.1: Use the appropriate formulas for weights and bias parameters during backward propagation.
            b_candidateState = self.calculate_new_state(self.tanh(self.tanh(self.cell[q]), derivative=True), b_hiddenState
                                                        , self.outputgate[q]) + b_updatecandidate
            # comes from the formula: ∂E /∂C of t = ∂E/∂h of t ⋅ o of t ⋅ tanh′(C of t) + ∂E/∂C of t+1 ⋅ f of t+1

            b_forgetState, b_wforget, b_bforget = self.calculate_totals(b_candidateState, self.cell[q - 1],
                                                                        self.sigmoid(self.forget[q], derivative=True),
                                                                        b_wforget, b_bforget, reshaped_input_q)
            # formula for forgetState: ∂E/∂f of t = (∂E/∂C~ of t ⋅ C of t-1) ⋅ σ′(f of t)
            # formula for b_wforget: ∂E/∂W of if = ∂E/ ∂f of t ⋅ x of t
            # formula for b_bforget: ∂E/∂b of if =  ∂E/∂f of t

            b_inputState, b_winput, b_binput = self.calculate_totals(b_candidateState, self.candidate[q],
                                                                     self.sigmoid(self.input_gates[q], derivative=True),
                                                                     b_winput, b_binput, reshaped_input_q)
            # formula for inputState: ∂E/∂i of t = (∂E/∂C~ of t ⋅ C~ of t) ⋅ σ′(i of t), where σ′ is sigmoid
            # formula for b_winput: ∂E/∂W of ii = ∂E/ ∂i of t ⋅ x of t
            # formula for b_binput: ∂E/∂b of ii =  ∂E/∂i of t

            b_candidateState, b_wcandidate, b_bcandidate = self.calculate_totals(b_candidateState, self.input_gates[q],
                                                                                 self.tanh(self.candidate[q],
                                                                                     derivative=True),
                                                                                 b_wcandidate, b_bcandidate,
                                                                                 reshaped_input_q)
            # formula for b_candidateState: ∂E/∂C~ of t = (∂E/∂C~ of t ⋅ i of t) ⋅ tanh′(C~t)
            # formula for b_wcandidateState: ∂E/∂W of ig = ∂E/ ∂C~ of t ⋅ x of t
            # formula for b_binput: ∂E/∂b of ig =  ∂E/∂C~ of t

            b_total_input = (np.dot(self.weights_forget, b_forgetState) + np.dot(self.weights_input, b_inputState) +
                             np.dot(self.weights_candidate, b_candidateState) + np.dot(self.weights_output,
                                                                                       b_outputState))
            # b = (W of hf ⋅ ∂E/∂f of t) + (W of hi ⋅ ∂E/∂i of t) + (W of hg ⋅ ∂E/∂C~ of t) + (W of ho ⋅ ∂E/∂o of t#0)

            b_updatehidden = b_total_input[:self.hidden_size, :]

            b_updatecandidate = self.forget[q] * b_candidateState

        for b_ in (b_wforget, b_bforget, b_winput, b_binput, b_wcandidate, b_bcandidate, b_woutput, b_boutput, b_wfinal,
                   b_bfinal):
            np.clip(b_, -1, 1, out=b_)

        # Achieving Objective 4.2.2: Update weights and biases using the computed gradients to fine-tune the model.
        desired_shape = self.weights_forget.shape
        self.weights_forget, self.bias_forget = self.new_weights_and_bias(desired_shape, b_wforget, b_bforget,
                                                                                 self.learning_rate,
                                                                                 self.weights_forget,
                                                                                 self.bias_forget)
        self.weights_input, self.bias_input = self.new_weights_and_bias(desired_shape, b_winput, b_binput,
                                                                        self.learning_rate, self.weights_input,
                                                                        self.bias_input)
        self.weights_candidate, self.bias_candidate = self.new_weights_and_bias(desired_shape, b_wcandidate,
                                                                                b_bcandidate, self.learning_rate,
                                                                                self.weights_candidate,
                                                                                self.bias_candidate)
        self.weights_output, self.bias_output = self.new_weights_and_bias(desired_shape, b_woutput, b_boutput,
                                                                          self.learning_rate, self.weights_output,
                                                                          self.bias_output)

        desired_shape = b_wfinal.shape

        self.weight_final, self.bias_final = self.new_weights_and_bias(desired_shape, b_wfinal, b_bfinal,
                                                                       self.learning_rate, self.weight_final,
                                                                       self.bias_final)

    def calculate_totals(self, b_candidateState, b_ofState, gate, b_weights, b_bias, reshapeb_inputStatenputs_q):
        # Function used for calculating weight and bias totals during backpropagation
        b_State = b_candidateState * b_ofState * gate
        b_weights += np.dot(b_State, reshapeb_inputStatenputs_q.T)
        b_bias += b_State

        return b_bias, b_weights, b_State

    def new_weights_and_bias(self, desired_shape, b_weights, b_bias, learning_rate, weights, bias):
        # function used for updating weights and bias to the variables initialised in initialised_weights
        padded_weights = np.pad(b_weights, (
            (0, desired_shape[0] - b_weights.shape[0]), (0, desired_shape[1] - b_weights.shape[1])), 'constant',
                                constant_values=(0)) # array used for updating weights and bias
        weights += padded_weights * learning_rate
        bias += b_bias * learning_rate
        # formula used to update the final weights and bias used earlier
        return weights, bias

    def calculate_new_state(self, cell, hidden, output):
        # function used for calculating new state during backpropagation
        b_newstate = cell * hidden * output

        return b_newstate

    def train(self, inputs, labels):
        # function used for training the LSTM
        inputs = [input_vals.flatten() for input_vals in inputs]
        print("Input Values (normalised data): ", inputs)
        for epoch in range(self.num_epochs):
            errors = []
            for q in range(len(inputs)):
                predictions = self.forward_propogation([inputs[q]])
                error = predictions[0] - labels[q]
                errors.append(error)

        self.backward_propogation(errors, [inputs[0]])

    def test(self, inputs, labels, mean, std):
        # Functions used for testing while using LSTM
        probabilities = self.forward_propogation([input_vals.flatten() for input_vals in inputs]) # array used for storing the probabilities
        print("Probabilities: ", probabilities)
        prediction = []
        errors= []
        # de-normalising the data
        # Lists used by sequentially storing prediction values
        for q in range(len(labels)):
            probabilities[q] = (probabilities[q] * 10)
        for q in range(len(labels)):
            prediction_value = (probabilities[q] * std + mean)
            prediction.append(prediction_value)
            actual_value = (labels[q] * std + mean)

            rmse = np.sqrt(np.mean((prediction_value - actual_value) ** 2))
            mape = np.mean(np.abs((prediction_value - actual_value) / actual_value)) * 100
            mae = np.mean(np.abs(prediction_value - actual_value))

            errors.append({'RMSE': rmse, 'MAPE': mape, 'MAE': mae})

        rmse_values = [entry['RMSE'] for entry in errors] # root mean-squared error
        mape_values = [entry['MAPE'] for entry in errors] # mean absolute percentage error
        mae_values = [entry['MAE'] for entry in errors] # mean absolute error

        median_rmse = np.median(rmse_values)
        median_mape = np.median(mape_values)
        median_mae = np.median(mae_values)

        # evaluation metrics used to assess the accuracy of the program
        print("Median RMSE:", median_rmse)
        print("Median MAPE:", median_mape)
        print("Median MAE:", median_mae)

        print(len)

        return prediction


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_stock_data():
    try:
        #  Acheiving Objective 9.2.1 : Provides autocomplete suggestions as users type in the search bar to assist in
        #  completing stock names or symbols, enhancing user experience by minimizing typing errors and improving search
        #  efficiency.
        # Read stock data from a CSV file
        all_stock_data = pd.read_csv(
            'C:\\Users\\kbrek\\PycharmProjects\\search_predictionsFTSE100\\FTSE100\\FTSE500_symbols.txt'
        )
        stock_codes = all_stock_data['Code'].tolist()  # use of lists
        stock_names = all_stock_data['Name'].tolist()  # use of lists

        return stock_codes, stock_names
    except Exception as e:
        # Exception handling: Print an error message if reading the CSV file fails
        print(f"Error reading CSV file: {e}")
        return [], []


@app.route('/home2')
def home2():
    # using a function to render the home page

    stock_codes, stock_names = read_stock_data()

    # URL paths for static stock images
    purple_stock_image = url_for('static', filename='CHI_stockchart.png')
    lightblue_stock_image = url_for('static', filename='ADM_Chart.png')
    darkred_stock_image = url_for('static', filename='ADIG_stockchart.png')
    darkblue_stock_image = url_for('static', filename='AAPL_stockchart.png')

    return render_template('index1.html', purple_url=purple_stock_image,
                           lightblue_url=lightblue_stock_image, darkred_url=darkred_stock_image,
                           darkblue_url=darkblue_stock_image, stock_codes=stock_codes,
                           stock_names=stock_names)


@app.route("/success")
def success(df, duration):
    # is returned when the historical data is successfully collected
    hidden_size = 30

    X, y, df_train, date_train, mean, std = data_prep(df, duration)

    # Achieving the Objective 6.1.1: Divide the historical dataset into training and testing sets and ensure the testing
    # set represents a realistic scenario by including recent data that the model has not seen during.
    X_train = X[:70]
    X_test = X[30:]

    y_train = y[:70]
    y_test = y[30:]

    # Using the LSTM class
    lstm = LSTM(input_columns=1, input_size=hidden_size, hidden_size=hidden_size, output_size=1, num_epochs=1_00,
                learning_rate=0.05)

    # Training the LSTM model
    lstm.train(X_train, y_train)

    # Testing the LSTM model
    predicition = lstm.test(X_test, y_test, mean, std)

    print(predicition)

    open_values = []

    for array in predicition:
        open_column = array[:, 0]
        open_values.extend(open_column)

    # Creating DataFrame for stock data
    forecasting_dates = pd.date_range(list(date_train)[-1], periods=len(open_values), freq='1d').tolist()
    dates = []
    for j in forecasting_dates:
        dates.append(j.date())

    df_final = pd.DataFrame(columns=['Date', 'Open'])
    df_final['Date'] = pd.to_datetime(dates)
    df_final['Open'] = open_values

    labels = df_final['Date'].dt.strftime('%Y-%m-%d')
    values = df_final['Open']

    labels = labels.to_list()  # instance of a list
    values = values.to_list()  # instance of a list

    return labels, values

def secure_filename(filename):
    return ''.join(c if c.isalnum() or c in {'_', '.'} else '_' for c in filename)

@app.route('//', methods=['POST'])
def upload_CSV():
    # this is a subroutine that parses the historical prices which the user to uploaded
    # Achieiving the Objective 5.1 & 5.2: Allows the user to input the historical stock data of any firm to their liking
    # into the website & I will store all historical data in a CSV file that is stored in a structured format.
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        duration = request.form["weeks"]
        filename = secure_filename(file.filename)
        file_path = os.path.join('C:\\Users\\kbrek\\PycharmProjects\\my_LSTM\\file_upload', filename)
        file.save(file_path)
        df = pd.read_csv(file_path)
        duration = int(duration) * 7
        labels, values = success(df, duration)
        return render_template('graph.html', labels=labels, values=values)

def save_stock_data_csv(stock_data, output_directory, symbol):
    # Saving stock data to CSV
    csv_filename = f'{symbol}_stock_data.csv'
    csv_path = os.path.join(output_directory, csv_filename)
    stock_data.to_csv(csv_path, index=False)
    return csv_path


def load_stock_data(symbol):
    # Retrieving the most recent stock data, if the library produces an erorr it should be able to load the data
    # stored already in the output directory in order for the data to be in the output directory the most recent
    # version should be saved there when the library does work
    stock_data = yf.Ticker(symbol).history(period='max').reset_index()

    if not stock_data.empty:
        symbol_adjusted = symbol + ".L" if not symbol.endswith('.') else symbol + "L"
        last_record_date = stock_data['Date'].iloc[len(stock_data) - 1]

        if last_record_date.year == 2024 and (len(stock_data) >= 365):
            # Save stock data to CSV
            save_stock_data_csv(stock_data, output_directory, symbol)

        else:
            stock_data = yf.Ticker(symbol_adjusted).history(period='max').reset_index()

            if not stock_data.empty:
                last_record_date = stock_data['Date'].iloc[0]
                symbol = symbol_adjusted

                if last_record_date.year == 2024 and (len(stock_data) >= 365):
                    # Save stock data to a CSV file
                    save_stock_data_csv(stock_data, output_directory, symbol_adjusted)

                else:
                    stock_data_path = os.path.join(output_directory, f'{symbol}_stock_data.csv')
                    # reads data from a csv file
                    if os.path.exists(stock_data_path):
                        stock_data = pd.read_csv(stock_data_path)

    if stock_data.empty:
        symbol_adjusted = symbol + ".L" if not symbol.endswith('.') else symbol + "L"
        stock_data = yf.Ticker(symbol_adjusted).history(period='max').reset_index()

        if not stock_data.empty:
            last_record_date = stock_data['Date'].iloc[0]
            symbol = symbol_adjusted

            if last_record_date.year == 2024 and (len(stock_data) >= 365):
                # Save stock data to a CSV file
                save_stock_data_csv(stock_data, output_directory, symbol)

        else:

            stock_data_path = os.path.join(output_directory, f'{symbol_adjusted}_stock_data.csv')

            if os.path.exists(stock_data_path):
                symbol = symbol_adjusted
                stock_data = pd.read_csv(stock_data_path)

            else:
                stock_data_path = os.path.join(output_directory, f'{symbol}_stock_data.csv')

                # read from a csv file
                if os.path.exists(stock_data_path):
                    stock_data = pd.read_csv(stock_data_path)

    print("Stock Data: ", stock_data)
    return stock_data


# Functions
def checkifsymbol(symbol):
    try:
        # Read symbol data from CSV
        all_data = pd.read_csv(
            'C:\\Users\\kbrek\\PycharmProjects\\search_predictionsFTSE100\\FTSE100\\FTSE500_symbols.txt')
        all_symbols = all_data['Code']

        symbol = symbol.strip()

        if symbol in all_symbols.values:
            return symbol

        matching_symbol = all_data.loc[all_data['Name'].str.strip() == symbol, 'Code'].values
        if len(matching_symbol) > 0:
            return matching_symbol[0]

    except Exception as e:
        # Error handling
        flash(f"Error checking symbol: {e}", 'error')
        return None


def plot_stock_graph(stock_data):
    # converts the stock data into something that can be turned into values that can be turned into a graph
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], utc=True)
    print("Dates: ", stock_data['Date'])
    labels = stock_data['Date'].dt.strftime('%Y-%m-%d')
    values = stock_data['Open']

    labels = labels.to_list()
    values = values.to_list()

    return labels, values


@app.route('/search', methods=['GET', 'POST'])
def search():
    # when a user searches for a stock, this fetches the histrorical data for that stock then produces a prediction graph
    # # Achieving Objective 9.1.2: Enables the user to enter the stock symbol or company name for their desired stock
    if request.method == 'POST':
        symbol = request.form['symbol']

        symbol = checkifsymbol(symbol)
        stock_data = load_stock_data(symbol)
        duration = 365

        labels, values = success(stock_data, duration)

        return render_template('graph.html', labels=labels, values=values)

    return render_template('index1.html', graph_html=None)


if __name__ == '__main__':
    app.run(port=5000, debug=True)