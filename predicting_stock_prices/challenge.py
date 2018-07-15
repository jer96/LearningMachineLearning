import tweepy
import csv
import numpy as np
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense

#Step 1 - Insert your API keys
consumer_key= ''
consumer_secret= ''
access_token=''
access_token_secret=''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#Step 2 - Search for your company name on Twitter

def search_for_tweets(term='google'):
	return api.search(term)

#Step 3 - Define a threshold for each sentiment to classify each 
#as positive or negative. If the majority of tweets you've collected are positive
#then use your neural network to predict a future price
def isPositiveSentiment(search, min_polarity=.7):
	tweets = len(search)
	pos_count = 0
	for tweet in search:
		analysis = TextBlob(tweet.text)
		pol = analysis.sentiment.polarity
		sub = analysis.sentiment.subjectivity
		# keep positive polarity with high subjectivty
		# or keep low subjectivity (highly factual information)
		if (pol >= min_polarity and sub >= .6) or (sub < .4):
			pos_count += 1
	return True if pos_count >= (tweets / 2) else False


def get_data(filename):
	#data collection
	dates = []
	prices = []
	
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	
	dates = [i for i in range(len(dates))]
	return dates, prices

#Step 5 reference your CSV file here
dates, prices = get_data('/Users/jeremiahbill/Desktop/ml/LearningMachineLearning/predicting_stock_prices/GOOG.csv')

#Step 6 In this function, build your neural network model using Keras, train it, then have it predict the price 
#on a given day. We'll later print the price out to terminal.
def predict_prices(dates, prices, x):
	length = len(dates)
	dates = [i for i in range(length)]
	dates = np.reshape(dates, (length, 1))
	prices = np.reshape(prices, (length, 1))

	split = .75
	split = int(length * split)
	
	X_train = dates[:split]
	y_train = prices[:split]

	X_test = dates[split:]
	y_test = dates[split:]
	
	model = Sequential()

	model.add(Dense(units=32, init='uniform', activation='relu', input_dim=1))
	model.add(Dense(units=32, init='uniform', activation='relu', input_dim=1))
	model.add(Dense(units=16, init='uniform', activation='relu'))
	model.add(Dense(units=1, init='uniform', activation='relu'))

	model.compile(loss='mean_squared_error',
			optimizer='adam',
            metrics=['accuracy'])
	
	model.fit(X_train, y_train, epochs=100, batch_size=3)

	loss_and_metrics = model.evaluate(X_test, y_test)
	print(loss_and_metrics)
	print(model.predict(X_test))


pos_sent = isPositiveSentiment(search_for_tweets())
if(pos_sent):
	predict_prices(dates, prices, 30)
else:
	print('unfortunately there is not positive sentiment or factual information available on this company so stock price will not be predicted')

# predicted_price = predict_price(dates, prices, 29)
# print(predicted_price)
