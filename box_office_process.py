import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
# import sklearn
# # import statsmodels.api as sm
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# from sklearn import metrics
# from sklearn.grid_search import GridSearchCV
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import confusion_matrix

# import seaborn as sns
# sns.set_style("whitegrid")
# sns.set_context("poster")
import warnings
warnings.filterwarnings('ignore')


# from sklearn import cross_validation
# from matplotlib import rcParams
# from bs4 import BeautifulSoup
# from pyquery import PyQuery as pq
from collections import defaultdict 
from imdb import IMDb
import pandas as pd
import _pickle as pickle
import io 
import time
import requests

ia = IMDb()
def get_mpaa(movieobj):
    try:
        mpaa = str(movieobj.data['mpaa']).split("Rated ", 1)[1].split(" ")[0]
    except:
        mpaa = np.nan
    return mpaa

BOdict = pickle.load(io.open('BOdict.p', 'rb'))

BOdf = pd.DataFrame(BOdict).transpose()

##Culling the dataset down to ensure we have non-null responses in our keys variables
limiteddf = BOdf.dropna(subset=['budget', 'season', 'mpaa', 'opening'])

## Ensuring that the number values are not in text format
limiteddf['gross'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
limiteddf['opening'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
limiteddf['opening theaters'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

##Replacing empty values
limiteddf.loc[limiteddf['opening']=='', 'opening']  = 0
limiteddf.loc[limiteddf['opening theaters']=='', 'opening theaters']  = 0

##Converting to float values for numerical variables
limiteddf['gross'] = limiteddf['gross'].astype(float)
limiteddf['opening'] = limiteddf['opening'].astype(float)
limiteddf['opening theaters'] = limiteddf['opening theaters'].astype(float)
limiteddf['budget'] = limiteddf['budget'].astype(float)
limiteddf['rating'] = limiteddf['rating'].astype(float)

##Converting to season (as necessary)
#limiteddf.loc[limiteddf['season']==0, 'season'] = 'Jan-May'
#limiteddf.loc[limiteddf['season']==1, 'season'] = 'June-Sep'
#limiteddf.loc[limiteddf['season']==2, 'season'] = 'Oct-Nov'
#limiteddf.loc[limiteddf['season']==3, 'season'] = 'Dec'

#Creating dummy variables for the various seasons
seasonlist = limiteddf.season.unique()
for season in seasonlist:
    limiteddf[season] = limiteddf['season']==season  

# Invoking a procedure similar to get_mpaa in order to process the MPAA rating
for i in limiteddf.index:
    try:
        limiteddf.loc[i, 'mpaa_new'] = limiteddf.loc[i, 'mpaa'].split("Rated ", 1)[1].split(" ")[0]
    except:
        limiteddf.loc[i, 'mpaa_new'] = 'PG-13'
limiteddf.loc[limiteddf['mpaa_new']=='PG-', 'mpaa_new'] = 'PG'
limiteddf.loc[limiteddf['mpaa_new']=='NC-17', 'mpaa_new'] = 'R'

#Creating dummy variables for the various MPAA Ratings
mpaalist = limiteddf.mpaa_new.unique()
for mpaa in mpaalist:
    limiteddf[mpaa] = limiteddf['mpaa_new']==mpaa
    
#Creating a list of prolific studios
studiodf = limiteddf.groupby('studio') 
studioslist = studiodf['title'].count()
studioslist.sort_values(ascending=False)

#Identifying the top-5 studios
limiteddf['prol_studio'] = False
for i in studioslist.index[:5]:
    limiteddf.loc[limiteddf['studio']==i,'prol_studio'] = True
    
#Identifying the next 5 top studios
limiteddf['Tier_2'] = False
for i in studioslist.index[6:12]:
    limiteddf.loc[limiteddf['studio']==i,'Tier_2'] = True

#Renaming the columns for use later
limiteddf.rename(columns={'opening theaters': 'opening_theaters', 'opening': 'opening_gross'}, inplace=True)

limiteddf.to_csv("data_box_office.csv")