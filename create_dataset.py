import pandas as pd
import numpy as np
import math

data = pd.read_csv("title.basics.tsv", sep="\t", low_memory=False)
ratings = pd.read_csv('data_ratigs.tsv', sep='\t', low_memory=False)

tropes = pd.read_csv('trope_dataset.csv',low_memory=False)

data = data[data['tconst'].isin(ratings['tconst'])]
 
print(data.shape)

data["averageRating"] = ratings["averageRating"]

data["numVotes"] = ratings["numVotes"]

print(data.shape)

movies = data['titleType'] == 'movie'

movie_data = data[movies]

movie_data = movie_data.drop(['titleType', 'isAdult', 'endYear', 'runtimeMinutes'], axis = 1)

movie_titles = list(movie_data['primaryTitle'])

for i in range(len(movie_titles)):
	movie_titles[i] = ''.join(e for e in movie_titles[i] if e.isalnum())

movie_data['primaryTitle'] = movie_titles

tropes.rename(columns={'Unnamed: 0': 'primaryTitle'},inplace=True)

same_movies = [] 

movie_data.sort_values(by="primaryTitle", ascending=True, inplace=True)

print("-------------------------")

for movie in tropes['primaryTitle']:
	if len(list(same_movies)) == 0:
		same_movies = movie_data['primaryTitle'] == movie
		if sum(same_movies) != 1:
			same_movies == []
	else:
		same = movie_data['primaryTitle'] == movie
		if sum(same) == 1:
			same_movies = same_movies | same

movie_data[same_movies].to_csv(r"same.csv")