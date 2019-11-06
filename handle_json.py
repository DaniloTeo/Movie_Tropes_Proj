import pandas as pd
import numpy as np
import json


with open('jasao.json') as data:
	dic = json.load(data)

tropes = []
movies = []
for movie in dic:
	movies.append(movie)
	for trope in dic[movie]:
		tropes.append(trope)

df = pd.DataFrame(index=np.unique(movies),columns=np.unique(tropes),dtype = bool)

df[:] = False

for movie in dic:
	for trope in dic[movie]:
		df[trope][movie] = True

print(df.shape)
# df.to_csv(r'trope_dataset.csv')