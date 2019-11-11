#%matplotlib inline 
from matplotlib import rcParams # special matplotlib argument for improved plots
from bs4 import BeautifulSoup
from collections import defaultdict 
from imdb import IMDb
from pyquery import PyQuery as pq
import re
import numpy as np
import pandas as pd
#import scipy.stats as stats
#import matplotlib.pyplot as plt
#import statsmodels.api as sm
import _pickle as pickle
# import seaborn as sns
# sns.set_style("whitegrid")
# sns.set_context("poster")

import io 
import time
import requests
#import sklearn
import warnings
warnings.filterwarnings('ignore')




years = range(1995,2018)
pages = range(1,9)
year_pagetxt = {}
for year in years: 
    pagestext = {}
    for page in pages: 
        r = requests.get("http://www.boxofficemojo.com/yearly/chart/?page=%s&view=releasedate&view2=domestic&yr=%s&p=.htm"%(page, year))
        pagestext[page] = r.text
        time.sleep(1)
    year_pagetxt[year] = pagestext





# This loop cycles through the data obtained on Box Office Mojo
## And placed in a dictionary for recall later when folding in IMDb information
movie_budget = defaultdict(list) 
for year in years: 
    for page in pages: 
        soup = BeautifulSoup(year_pagetxt[year][page], "html.parser")
        rows = soup.find_all("font", attrs={'size':'2'})
           
        start = 10 
        for i in range(start,len(rows)-2):
            t = rows[i].get_text()
            if unicode('Summary of') in t: 
                break
            elif (i-start) % 9 == 0: 
                movie_budget['rank'].append(t)
            elif (i-start) % 9 == 1: 
                movie_budget['year'].append(year)
                r = '('+str(year)
                if unicode(r) in t: 
                    j = t.index(unicode(r))
                    movie_budget['title'].append(t[:j])
                else: 
                    movie_budget['title'].append(t)
            elif (i-start) % 9 == 2: 
                movie_budget['studio'].append(t)
            elif (i-start) % 9 == 3: 
                movie_budget['gross'].append(t)
            elif (i-start) % 9 == 4: 
                movie_budget['gross theaters'].append(t)
            elif (i-start) % 9 == 5: 
                movie_budget['opening'].append(t)
            elif (i-start) % 9 == 6: 
                movie_budget['opening theaters'].append(t)
            elif (i-start) % 9 == 7: 
                movie_budget['open'].append(t)
            elif (i-start) % 9 == 8: 
                movie_budget['close'].append(t)




#the-numbers.com is a separate movie website that has the budgets for most major movies.
r_numbers = requests.get("http://www.the-numbers.com/movie/budgets/all")

d_=pq(r_numbers.text)
d_tables=pq(d_('table'))
rows = pq(d_tables[0])('tr')



#This processes the budget information and passes it to a new dictionary
budget = defaultdict(list)
for j in range(1,len(rows)):
    dat = pq(rows[j])('td')
    for i in range(len(dat)):
        if i % 6 == 1:
            aux = re.split('/|, ',pq(dat[i])('a').text())[-1]
            aux = int(aux)
            budget['year'].append(aux)
        elif i % 6 == 2:
            t = pq(dat[i])('a').text()
            if 'Birdman' in t:
                budget['title'].append(t.split(' or ')[0])
            else:
                budget['title'].append(t)
        elif i % 6 == 3: 
            budget['budget'].append(pq(dat[i]).text())



def find_movie(title, year,  movie_list):
    """
    find_movie: given the movie title (type: string) that you desire, 
    the year (type:int) that your desired movie came out, and a list of 
    movies (type:list containing imdbpy movie objects), this functon will 
    return the movie object that has a title that best matches yours. 
    If there are no plausible matches, it will return None. 
    """
    # find movies that came out in the same year                                                                                                                                    
    year_list = []
    for movie in movie_list:
        try:
            if movie.data['year'] == int(year):
                year_list.append(movie)
        except:
            continue
    # if the years do not match, there is no match                                                                                                                                  
    if len(year_list) < 1:
        return None
    else:
        # process the desired title                                                                                                                                                         
        sorted_title = "".join(sorted(title)).replace(" ", "")
        len_sorted_title = len(sorted_title)
        # check whether movies that came out in the same year contain the same letters                                                                                                                      
        counts = [0]*len(year_list)
        for j in range(len(year_list)):
            # process each movie title 
            movie_title = year_list[j]['title']
            sorted_movie_title = "".join(sorted(movie_title)).replace(" ", "")
            if len_sorted_title == len(sorted_movie_title):
                # if the title cannot be converted to a string it is not the correct title                                                                                                                                       
                try:
                    sorted_movie_title = str(sorted_movie_title)
                except:
                    continue
                for i in range(len_sorted_title):
                    if sorted_title[i].lower() == sorted_movie_title[i].lower():
                        counts[j] += 1
            else:
                continue
        
        if max(counts) <= len_sorted_title: 
            k = counts.index(max(counts))
        else: 
            k = counts.index(len_sorted_title)
        if len(year_list) >= 1:
            return year_list[k]
        else:
            return None



# instantiate an IMDB object 
ia = IMDb(accessSystem='http')

BOmissingmovies = [] # Tracks movies that we cannot find a match for 
BOdict = {} # Contains movie information 
movienumber = len(movie_budget['year'])

for i in range(movienumber):
    movieobj = None
    # Need to process the row-level information out of BOdf in order to get the movie objects
    movieobj = ia.search_movie(movie_budget['title'][i])
    
    #Handling cases where we haven't found the movie or have multiples
    if movieobj is None or len(movieobj)>1:
        potential_movie_titles = ia.search_movie(movie_budget['title'][i])
        movieobj = find_movie(movie_budget['title'][i], movie_budget['year'][i], potential_movie_titles) # find the movie
        if type(movieobj) == list: 
            movieobj = movieobj[0]
    
    if movieobj is not None and not (type(movieobj) == list and len(movieobj) == 0):
        ## Get movie id ##
        if type(movieobj) == list: 
            movieobj = movieobj[0]
        ia.update(movieobj)    
        movid = movieobj.movieID
        ## Populate dictionary, main key is movie id ##
        BOdict[movid] = {}
        # "title": title of movie
        BOdict[movid]['title'] = movie_budget['title'][i]
        # "gross": Domestic Gross Revenue for the movie
        BOdict[movid]['gross'] = movie_budget['gross'][i]
        # "opening": Opening Weekend Revenue
        BOdict[movid]['opening'] = movie_budget['opening'][i]
        # "Rank": Final Rank of Revenue for the Year
        BOdict[movid]['rank'] = movie_budget['rank'][i]
        # "studio": Studio that created the movie
        BOdict[movid]['studio'] = movie_budget['studio'][i] 
        # "open": Date that the movie opened on
        BOdict[movid]['open'] = movie_budget['open'][i]
        # "close": Date the movie closed domestically
        BOdict[movid]['close'] = movie_budget['close'][i]
        # "opening theaters": Number of theaters that the movie opened in
        BOdict[movid]['opening theaters'] = movie_budget['opening theaters'][i]
        # "year": Year that the movie was released
        BOdict[movid]['year'] = movie_budget['year'][i]
        try:
            BOdict[movid]['rating'] = movieobj['rating']
        except: 
            BOdict[movid]['rating'] = None
        try:
            BOdict[movid]['mpaa'] = movieobj['mpaa']
        except: 
            BOdict[movid]['mpaa'] = None
        try:
            BOdict[movid]['director'] = None
        except: 
            BOdict[movid]['director'] = None
        try:
            BOdict[movid]['cast'] = None
        except:
            BOdict[movid]['cast'] = None
    else:
        BOmissingmovies.append((i, movie_budget['title'][i], movie_budget['year'][i]))






# Add feature describing which season the movie was released in 

## Our Seasons do not follow the typical calendar seasons, as movie timetables typically
## rely on specific months of the year

## For example, June-September are usually the 'summer blockbuster' months
## Similarly, December is treated on its own since the Holiday season is an important
## time for most movies

for k in BOdict.keys():
    open_date = BOdict[k]['open']
    if open_date:
        month = int(open_date.split('/')[0])
        if month <= 5 and month >= 1:
            BOdict[k]['season'] = 'Jan-May'
        elif month <= 9 and month >= 6:
            BOdict[k]['season'] = 'June-Sep'
        elif month <= 11 and month >= 10:
            BOdict[k]['season'] = 'Oct-Nov'
        elif month == 12: 
            BOdict[k]['season'] = 'Dec'
        else:
            BOdict[k]['season'] = None





# Add feature describing the movies budget 
c = 0
for k in BOdict.keys():
    for i in range(len(budget['title'])):
        if sorted(unicode(budget['title'][i].lower().replace(' ', ''))) == sorted(BOdict[k]['title'].lower().replace(' ','')):
            BOdict[k]['budget'] = float(budget['budget'][i][1:].replace(',',''))
            break
        else:
            BOdict[k]['budget'] = None




# Since this took a long time to run, we save it here to ensure it is available for later use. 
pickle.dump(BOdict, io.open('BOdict.p', 'wb'))