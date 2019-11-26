import pandas as pd
import numpy as np

def classify_value(val, dic):
  if (dic["Low"][0] <= val < dic["Low"][1]):
    #print("Low")
    return 0
  if (dic["Low Medium"][0] <= val < dic["Low Medium"][1]):
    #print("LM")
    return 1
  if (dic["Medium"][0] <= val < dic["Medium"][1]):
    #print("M")
    return 2
  if (dic["Medium High"][0] <= val < dic["Medium High"][1]):
    #print("MH")
    return 3
  if (dic["High"][0] <= val <= dic["High"][1]):
    #print("H")
    return 4


def classify_column(dic, df):
  classified_df = df.copy()
  classified_df.astype(int)
  #print(classified_df.head())
  for row in range(len(df.index)):
    classified_df.iloc[row] = classify_value(df.iloc[row], dic)

  #print(classified_df.head())
  return classified_df

def discretize_column(df):
    #print("entrou")
    # redefine dataframe values and create safety copy
    start = df.min(); end = df.max()
    new_df = df.copy()
    new_df.astype(int)

    # define parameters for the interval definition
    total_len = end - start
    #print("start: " + str(start)+" end: " + str(end))
    sub_len = total_len/5
    
    # define categories
    dic = {"Low": [], "Low Medium": [],"Medium": [], "Medium High": [],"High": []}
    
    # define range to each category
    curr_start = start
    for key in dic.keys():
      # define the interval for the current category
      dic[key].append(curr_start); dic[key].append(curr_start + sub_len)
      # update interval
      curr_start += sub_len
    
    new_df = classify_column(dic, df)
    #print(type(new_df));print(new_df.shape)
    #print(new_df)
    return new_df


def discretize_to_int(df):
  aux = df.copy()
  unq = np.unique(aux)
  #print(unq)
  #print(len(unq))
  #print("*****************************")
  for u in range(len(unq)):
    ids = np.where(df == unq[u])
    for i in ids:
      aux[i] = u
  #print("---------------------------")
 # print(aux)
  return aux

def discretize(df, columns=['budget', 'gross', 'opening_gross', 'opening_theaters', 'rating']):
  disc = df.copy()
  for col in columns:
    disc[col] = discretize_column(df[col])
  for col in ['season', 'studio', 'mpaa_new']:
    disc[col] = discretize_to_int(df[col])
  return disc



#main
df = pd.read_csv("data_box_office.csv")
clean = df.drop(['cast','close', 'director', 'mpaa', 'open', 'rank', 'June-Sep', 'Oct-Nov', 'Jan-May', 'Dec', 'PG', 'R', 'PG-13', 'prol_studio', 'Tier_2'], axis=1)
disc = discretize(clean)
disc.to_csv('data_discretizado_native.csv')
