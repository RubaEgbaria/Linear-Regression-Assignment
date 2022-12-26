import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as mat
import seaborn as sb
import matplotlib.image as mpimg
import IPython.display as display
from sklearn import linear_model
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        mat.tight_layout()
    mat.savefig(path, format=fig_extension, dpi=resolution)


def hypothesis(X, theta):
    return np.dot(X, theta)


# function to compute gradient of error function w.r.t. theta
def gradient(X, y, theta):
    h = hypothesis(X, theta)
    grad = np.dot(X.transpose(), (h - y))
    return grad


# function to compute the error for current values of theta
def cost(X, y, theta):
    h = hypothesis(X, theta)
    J = np.dot((h - y).transpose(), (h - y))
    J /= 2
    return J[0]


# function to create a list containing mini-batches
def create_mini_batches(X, y, batch_size):
    mini_batches = []
    data = np.hstack((X, y))
    np.random.shuffle(data)
    n_minibatches = data.shape[0] // batch_size
    i = 0

    for i in range(n_minibatches + 1):
        mini_batch = data[i * batch_size:(i + 1) * batch_size, :]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i * batch_size:data.shape[0]]
        X_mini = mini_batch[:, :-1]
        Y_mini = mini_batch[:, -1].reshape((-1, 1))
        mini_batches.append((X_mini, Y_mini))
    return mini_batches


# function to perform mini-batch gradient descent
def gradientDescent(X, y, learning_rate=0.001, batch_size=32):
    theta = np.zeros((X.shape[1], 1))
    error_list = []
    max_iters = 3
    for itr in range(max_iters):
        mini_batches = create_mini_batches(X, y, batch_size)
        for mini_batch in mini_batches:
            X_mini, y_mini = mini_batch
            theta = theta - learning_rate * gradient(X_mini, y_mini, theta)
            error_list.append(cost(X_mini, y_mini, theta))

    return theta, error_list


# 1-Load housing sales data using pandas read_csv()
housing = pd.read_csv("kc_house_data.csv")

# 2-List the following about data  a.Number of features b.The features list and their datatype c.The number of samples
housing.head()
housing.info()
print("Samples : ", housing.size)

# 3-Compute the mean of House Prices Extract the price column of the sales Series a.prices=sales['price'] b.avg_price_2=prices.mean()
prices = housing['price']
print("Mean : ", prices.mean())

#4) Compute the sum of the squares of the House Prices in the data set using pandas function.
prices = pd.DataFrame(prices)
sq_prices = np.power((prices),2)
print("Sum squeres : ", sq_prices.sum())

#5) Histogram sqft_living and the prices features. Your plots must be labeled.
print("blah blah ..")
housing['sqft_living'].plot.hist()
mat.title("sqft_living feature")
mat.xlabel("sqft_living")
mat.show()
housing['price'].plot.hist()
mat.title("price feature")
mat.xlabel("price")
mat.show()

#6) Display information about the data set using pandas describe method
print(housing.describe())

#7) Create a scatter plot of the price vs sqft_living (hint: use sales.plot(… ) see chapter 2).
#Use alpha = .1 to make your plot show density as well.
housing.plot.scatter(x = "price" , y = "sqft_living" , alpha = 0.1 )
mat.show()

#8) Using the pandas corr() function, which two features the price feature is correlated to the most? How much?
c = housing.corr()
sb.heatmap(c)
mat.show()
for i in range(c.shape[0]):
    for j in range(i+1, c.shape[0]):
        if c.iloc[i,j] >= 0.8:
            print("Correlation.maxbetween those {}  {} is :  {} ".format(c.index[i] , c.columns[j] , c.iloc[i,j]))

#9) Is there any missing data?
a = 0
print(housing.isnull().sum())

#10 Split the data into training and testing sets using sklearn train_test_split function (hint
#chapter 2 ). Fix the random seed so you always split the same way.
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print("after spliting train set : " , train_set)
print("after spliting test set : " , test_set)

#11 Build a regression model for predicting price based on sqft_living. Rembember that to
#train using the train_data. Use LinearRegression
trainData1 = train_set[["sqft_living"]]
trainData1 = trainData1.reset_index( drop = True )
trainData2 = train_set[["price"]]
trainData2 = trainData2.reset_index( drop = True )
testData1 = test_set[["sqft_living"]]
testData1 = testData1.reset_index( drop = True )
testData2 = test_set[["price"]]
testData2 = testData2.reset_index( drop = True )
model = linear_model.LinearRegression()
model.fit(trainData1, trainData2)
model.coef_
model.intercept_

#12  What are the model parameters
print("R squared training = ",model.score(trainData1, trainData2))
print("RMSE = ", mean_squared_error(testData2, model.predict(testData1)))

#13  Create a scatter plot of training data price feature vs sqft_living feature and then
#superimpose in red the model on top

mat.figure(figsize = (12, 10))
mat.scatter(trainData1, trainData2, color = 'darkgreen', label = 'Training data')
mat.plot(trainData1, trainData2, "b.")
mat.plot(trainData1, model.predict(trainData1), color = 'red', label= 'Predicted Regression line')
mat.xlabel('sqft_living', fontsize=18)
mat.ylabel('price', rotation=0, fontsize=18)
mat.legend()
save_fig("sqft_living_price")
mat.show()

#14 Repeat 11-14 using mini-batch gradient decent SGDRegressor (See chapter 4 for how to do that)
#11 Build a regression model for predicting price based on sqft_living. Rembember that to
#train using the train_data. Use LinearRegression
trainData1 = train_set[["sqft_living"]]
trainData1 = trainData1.reset_index( drop = True )
trainData2 = train_set[["price"]]
trainData2 = trainData2.reset_index( drop = True )
testData1 = test_set[["sqft_living"]]
testData1 = testData1.reset_index( drop = True )
testData2 = test_set[["price"]]
testData2 = testData2.reset_index( drop = True )
theta, error_list = gradientDescent(trainData1, trainData2)
model = linear_model.LinearRegression()
model.fit(trainData1, trainData2)
model.coef_
model.intercept_
#12  What are the model parameters
print("R squared training = ",model.score(trainData1, trainData2))
print("RMSE = ", mean_squared_error(testData2, model.predict(testData1)))
print("Bias = ", theta[0])
print("Coefficients = ", theta[1:])

#13  Create a scatter plot of training data price feature vs sqft_living feature and then
#superimpose in red the model on top
mat.plot(error_list)
mat.xlabel("sqft_living")
mat.ylabel("price")
mat.show()


#15 Repeat 11-13 using the most two features that correlate with the house price
#11 Build a regression model for predicting price based on sqft_living. Rembember that to
#train using the train_data. Use LinearRegression
trainData1 = train_set[["bedrooms"]]
trainData1 = trainData1.reset_index( drop = True )
trainData2 = train_set[["price"]]
trainData2 = trainData2.reset_index( drop = True )
testData1 = test_set[["bedrooms"]]
testData1 = testData1.reset_index( drop = True )
testData2 = test_set[["price"]]
testData2 = testData2.reset_index( drop = True )
model = linear_model.LinearRegression()
model.fit(trainData1, trainData2)
model.coef_
model.intercept_

#12  What are the model parameters
print("R squared training = ",model.score(trainData1, trainData2))
print("RMSE = ", mean_squared_error(testData2, model.predict(testData1)))

#13  Create a scatter plot of training data price feature vs sqft_living feature and then
#superimpose in red the model on top
mat.figure(figsize = (10, 10))
mat.scatter(trainData1, trainData2, color = 'darkgreen', label = 'Training data')
mat.plot(trainData1, trainData2, "b.")
mat.plot(trainData1, model.predict(trainData1), color = 'red', label= 'Predicted Regression line')
mat.xlabel('bedrooms', fontsize=18)
mat.ylabel('price', rotation=0, fontsize=18)
mat.legend()
save_fig("bedrooms_price")
mat.show()


print(" Done.. ")