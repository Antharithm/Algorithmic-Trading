# High Frequency Trading Algorithm - Optional 

![[cover.jpg]](Images/cover.jpg)

Due to the volatility in investment performance of human portfolio managers, many investment firms rely on trading algorithms to produce consistent investment performance capable of outperforming the markets. One such investing strategy is the combination of Machine Learning Algorithms and High Frequency Algorithms. This strategy has become very popular and requires skills from multiple of the previous units, such as automatically retrieving data used to make investment decisions, training a model, and building an algorithm to execute the trading.

You have been tasked by the investment firm Renaissance High Frequency Trading (RHFT) to develop such an algorithm. RHFT wants the algorithm to be based on stock market data for `FB`, `AMZN`, `AAPL`, `NFLX`, `GOOGL`, `MSFT`, and `TSLA` at the minute level. It should conduct buys and sells every minute based on 1 min, 5 min, and 10 min Momentum. The CIO asked you to choose the Machine Learning Algorithm best suited for this task and wants you to execute the trades via Alpaca's API.

You will need to:

1. [Prepare historical return data for training and testing (optional)](#part-1-prepare-the-data-for-training-and-testing-optional)
2. [Compare prediction performance of multiple ML-Algorithms](#part-2-train-and-compare-multiple-Machine-Learning-Algorithms)
3. [Implement a fully functional trading algorithm that buys and sells the stocks selected by the Model](#part-3-implement-the-strongest-model-using-alpaca-api)

NOTE: If you choose do [Part 1](#part-1-prepare-the-data-for-training-and-testing-optional), you can ignore the provided csv-file and use the optional notebook. The regular notebook begins at [Part 2](#train-and-compare-multiple-Machine-Learning-Algorithms).

- - -

### Files

[Starter Notebook](Starter_Code/high_frequency_trading_algo.ipynb)

[Starter Notebook (Optional Section)](Starter_Code/high_frequency_trading_algo_optional.ipynb)

[Starter CSV file](Starter_Code/returns.csv)

- - -

## Instructions

To begin, create an `env` file to store your Alpaca credentials using `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` as the variable names. Store this file in the same folder as your starter notebook.

### Part 1: Prepare the data for training and testing (optional)

Open the starter notebook.  The code for this section has been provided, however the steps are still outlined below to allow for understanding. The code contains functions to aquire and clean data, compute returns and create a final cleaned DataFrame to hold momentums.  

#### Initial Set-Up:

1. You should have already created an `env` file to store your Alpaca credentials using `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` as the variable names.
2. Using the starter notebook, run the provided code cells to:
    * Load your environment variables.
    * Generate your Alpaca API object, specifying use of the paper trading account with the base url "https://paper-api.alpaca.markets".


#### Data Generatation

1. Create a ticker list, beginning and end dates, and timeframe interval.
2. Ping the Alpaca API for the data and store it in a DataFrame called `prices` by using the `get_barset` function combined with the `df` method from the Alpaca Trade SDK.
3. Store only the close prices from the `prices` DataFrame in a new DataFrame called `df_closing_prices`, then view the head and tail to confirm the following:
    * The first price for each stock on the open is at 9:30 Eastern Time.
    * The last price for the day on the close is at 3:59 pm Eastern Time.
4. When viewing the head and tail, you'll notice several `NaN` values.
    * Alpaca reports `NaN` for minutes without any trades occuring as missing.
    * These values are removed using Panda's `ffill()` function to "forward fill", or replace, those prices with the previous values (since the price has not changed).

#### Computing Returns

1. Compute the percentage change values for 1 minute as follows:
    * Create a variable called `forecast` to hold the forecast, in this case `1` for 1 minute.
    * Use the `pct_change` function, passing in the `forecast`, on the `df_closing_prices` DataFrame, storing the newly generated DataFrame in a variable called `returns`.
    * Convert the `returns` DataFrame to show forward returns by passing `-(forecast)` into the `shift` function.
2. Convert the DataFrame into long form for merging later using `unstack` and `reset_index`.
3. Compute the 1, 5, and 10 minute momentums that will be used to predict the forward returns, then merge them with the forward returns as follows:
    * Create the list of moments: `list_of_momentums = [1,5,10]`
    * Write a for-loop to loop through the `list_of_momentums`, applying them to `pct_change` with the `df_closing_price` with each iteration.
    * With each loop, the temporary DataFrame, `returns_temp` will need to be prepped with `unstack` and `reset_index`, then merged with the original `returns` DataFrame.
    * Complete this step by dropping the null values from `returns` and creating a multi-index based on date and ticker.


### Part 2: Train and compare multiple Machine Learning Algorithms

In this section, you'll train each of the requested algorithms and compare performance. Be sure to use the same parameters and training steps for each model. This is necessary to compare each model accurately.

#### Preprocessing Data

Using the `results` DataFrame from part one, you'll preprocess your data to make it ready for machine learning.

1. Generate your feature data (`X`) and target data (`y`):
    * Create a dataframe `X` that contains all the columns from the returns dataframe that will be used to predict `F_1_m_returns`.
    * Create a variable, called `y`, that is equal to 1 if `F_1_m_returns` is larger than 0. This will be our target variable
2. Use the train_test_split library to split the dataset into a training and testing dataset, with 70% used for testing
    * Set the shuffle parameter to `False` to use only the first 70% of the data for training (this prevents look ahead bias).
    * Make sure you have these 4 variables: `X_train`, `X_test`, `y_train`, `y_test` 
3. Use the `Counter` function to test the distribution of the data. The result of `Counter({1: 668, 0: 1194})` reveals the data is indeed unbalanced.
4. Balance the dataset with the Oversampler libary, setting `random state= 1`.
5. Test the distribution once again with `Counter`. The new result of `Counter({1: 1194, 0: 1194})` shows the data is now balanced.

#### Machine Learning

With the data preprocessed, you can now train the various algorithms and evaluate them based on precision using the `classification_report` function from the sklearn library.

1. The first cells in this section provide an example of how to fit and train your model using the `LogisticRegression` model from sklearn:
    * Import selected model.
    * Instantiate model object.
    * Fit the model to the resampled data - `X_resampled` and `y_resampled`.
    * Predict the model using `X_test`.
    * Print the classification report.

 2. Use the same approach as above to train and test the following ML Algorithms:
    * [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    * [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
    * [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
    * [XGBClassifier](https://xgboost.readthedocs.io/en/latest/python/python_api.html)

#### Evaluate the performance of each model

1. Use the classification reports to answer the following questions in a markdown cell:

    > Which model produces the highest Accuracy?
    >
    > Which model produces the highest performance over time?
    >
    > Which model produces the highest Sharpe Ratio?

2. Using the classification report for each model, choose the model with the highest precision for use in your algo-trading program.
3. Save the selected model with the `joblib` libary to avoid retraining every time you wish to use it.



### Part 3: Implement the strongest model using Alpaca API

In this final section, you'll create your algo-trading program by pulling live data at the minute frequency and ensuring that the model is buying the selected stocks. If you need a refresher on methods available using the Alpaca SDK, you can view their docs [here](https://github.com/alpacahq/alpaca-trade-api-python/).

#### Develop the Algorithm

1. Use the provided code to ping the Alpaca API and create the DataFrame needed to feed data into the model.
    * This code will also store the correct feature data in `X` for later use.
2. Using `joblib`, load the chosen model.
3. Use the model file to make predicttions:
    * Use `predict` on `X` and save this as `y_pred`.
    * Convert `y_pred` to a DataFrame, setting the index to the index of `X`.
    * Rename the column 0 to 'buy', be sure to set `inplace =True`.
4. Filter the stocks where 'buy' is equal to 1, saving the filter as `y_pred`.
5. Using the `y_pred` filter, create a dictionary called `buy_dict` and assign 'n' to each Ticker (key value) as a placeholder.
6. Obtain the total available equity in your account from the Alpaca API and store in a variable called `total_capital`. You will split the capital equally between all selected stocks per the CIO's request.
7. Use a for-loop to iterate through `buy_dict` to determine the number stocks you need to buy for each ticker.
8. Cancel all previous orders in the Alpaca API (so you don't buy more than intended) and sell all currently held stocks to close all positions.
9. Iterate through `buy_dict` and send a buy order for each ticker with their corresponding number of shares.



#### Automate the Algorithm
To automate the algorithm, you'll combine all the steps above into one function that be executed automatically with Python's "schedule" module. For more information on this module you can view the docs [here](https://schedule.readthedocs.io/en/stable/).

1. Make a function called `trade()` that incorporates all of the steps above. **Note:** The data cleaning and calculations section from earlier has already been incorpoated. Your task is to complete the function starting where you see `# YOUR CODE HERE`. 
2. Import Python's schedule module.
3. Use the schedule module to automate the algorithm:
    * Clear the schedule with `.clear()`.
    * Define a schedule to run the trade function every minute at 5 seconds past the minute mark (e.g. `10:31:05`).
    * Use the Alpaca API to check whether the market is open.
    * Use the `run_pending()` function inside schedule to execute the schedule you defined while the market is open.

- - -

### Resources

[Comparing Machine Learning Algorithm Performance](https://keras.io/getting-started/sequential-model-guide/)

[Creating Stock Market Data for ML Algorithms](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)


- - -

### Hints and Considerations

Experiment with the model architecture and parameters to see which provides the best results, but be sure to use the same architecture and parameters when comparing each model.


- - -

### Submission

* Complete the starter Jupyter Notebook for the homework and host the notebook on GitHub.

* Include a README.md that summarizes your homework and include this report in your GitHub repository.

* Submit the link to your GitHub project to Bootcamp Spot.

- - -

© 2021 Trilogy Education Services, a 2U, Inc. brand. All Rights Reserved.
