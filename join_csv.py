import pandas as pd 
import numpy as np 

jason = pd.read_csv('trope_dataset.csv',low_memory=False)
movie_list = pd.read_csv('same.csv')
print("loaded")
print("sizeof trope table: " + str(jason.shape))
print("sizeof movie table: " + str(movie_list.shape))

same_movies = []

for movie in movie_list['primaryTitle']:
	if len(list(same_movies)) == 0:
		same_movies = jason['Unnamed: 0'] == movie
		if sum(same_movies) != 1:
			same_movies == []
	else:
		same = jason['Unnamed: 0'] == movie
		if sum(same) == 1:
			same_movies = same_movies | same
print("filtered")
jason = jason[same_movies]

print("sizeof trope table: " + str(jason.shape))

jason["averageRating"] = movie_list["averageRating"]
jason["numVotes"] = movie_list["numVotes"]

print("appended")

print("sizeof trope table: " + str(jason.shape))

aux = jason[jason['Unnamed: 0'] == "AmericanNinja"]
print(aux)

column_list = jason.columns[2:-2]

for col in column_list:
	if jason[col].any() == False:
		jason = jason.drop(columns=[col])

print(jason.shape)

# jason.to_csv("main_dataset.csv")
