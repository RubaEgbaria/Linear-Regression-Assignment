<h3> In this assignment you will use data on house sales in King County (USA), to predict house 
prices using simple (one feature) linear regression. Complete the following tasks for this 
assignment. Note, a second part for this assignment will follow.</h3>

<ol> 
<li>Load housing sales data using pandas read_csv()</li>
<li>List the following information about the data </li>
<ul>
<li>Number of features </li>
<li>The features list and their datatype </li>
<li>The number of samples </li>
</ul>
<li>Compute the mean of the House Prices in the data set. Extract the price column of the sales Series </li>
<ul>
<li>prices = sales['price'] </li>
<li>avg_price_2 = prices.mean()</li>
</ul>
<li>Compute the sum of the squares of the House Prices in the data set using pandas function. </li>
<li>Histogram sqft_living and the prices features. Your plots must be labeled.</li>
<li>Display information about the data set using pandas describe method</li>
<li>Create a scatter plot of the price vs sqft_living (hint: use sales.plot(… ) see chapter 2). Use alpha = .1 to make your plot show density as well.
<li>Using the pandas corr() function, which two features the price feature is correlated to the most? How much?</li>
<li>Is there any missing data?</li>
<li>Split the data into training and testing sets using sklearn train_test_split function (hint chapter 2 ). Fix the random seed so you always split the same way.</li>
<li>Build a regression model for predicting price based on sqft_living. Rembember that to train using the train_data. Use LinearRegression </li>
<li>What are the model parameters </li>
<li>Create a scatter plot of training data price feature vs sqft_living feature and then superimpose in red the model on top.</li>
<li>Repeat 11-14 using mini-batch gradient decent SGDRegressor (See chapter 4 for how to do that)</li>
<li>Repeat 11-13 using the most two features that correlate with the house price.</li>
</ol>
