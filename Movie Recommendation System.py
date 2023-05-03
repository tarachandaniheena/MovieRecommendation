#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd
import numpy as np
df1 = pd.read_csv("tmdb_5000_credits.csv")
df2 = pd.read_csv("tmdb_5000_movies.csv")


# In[102]:


# renaming movie id column to perform join with the second csv file
df1.columns = ['id','title','cast','crew']
df2 = df2.merge(df1, on='id')


# ## Demographic Filtering

# Demographic filtering is a generalized recommendation system based on the demographics of the users. This is a very simple recommendation based on the movie rating.
# 
# In order to use demographic filtering, we have to assign a score to the movie in order to recommend the best oness to the audience. We have the rating column available in dataset but its not weighted.

# Using IMDB's weighted rating 

# In[104]:


#C is mean vote across the whole report
C = df2['vote_average'].mean()
C


# Mean rating of all the movie across is approx 6 on the scake of 10. Next step is to determine approporiate value of m, the minimum votes requried for a movie to be listed in the chart.
# 
# Using 90th percentile as a cutoff, meaning a movie should have more number of votes than at least 90% of the movies in the list.

# In[105]:


m = df2['vote_count'].quantile(0.9)
m


# In[106]:


# filter out the movies that qualify for the chart
q_movies = df2.copy().loc[df2['vote_count'] >= m]
q_movies.shape


# 481 movies qualify to be in the list. Now we have to calculate metric for each movie using weighted_rating and define a new feature called score.

# In[107]:


def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m)*R) + (m/(v+m)*C)


# In[108]:


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)


# In[110]:


# Sort top 10 movies based on the score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

q_movies[['title_x','vote_count','vote_average','score']].head(10)


# ## Content Based Filtering

# Similarity amongst the content is identified to find similarity with other movies.

# In[111]:


# Plot description based Recommender
df2['overview'].head()


# Using TF-IDF to convert text into vectors to find the important words

# In[112]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = 'english')
df2['overview'] = df2['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(df2['overview'])
tfidf_matrix.shape


# 20000 words were used to describe 4800 movies in the dataset.

# We have to find a similairty score in the matrix using cosine similarity and linear_kernel

# In[113]:


from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# Define a function that takes move title as an input and outputs 10 similar movie titles. Given that cosine matrix contains index, we have to do reverse mapping from title to index.

# In[114]:


indices = pd.Series(df2.index, index = df2['title_x']).drop_duplicates()


# In[115]:


indicesThe Dark Knight Rises


# In[116]:


def get_recommendations(title_x, cosine_sim = cosine_sim):
     #get index of the movies that matches the title
        idx  = indices[title_x]
        
        # Get the pairwise similarity scores of all movies with respect to that movie
        sim_scores  = list(enumerate(cosine_sim[idx]))
        
        # Sort the movies based on the similarity score
        sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse=True)
        
        # get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]
        
        movie_indices = [i[0] for i in sim_scores]
        
        # reurn the top most 10 similar movies
        return df2['title_x'].iloc[movie_indices]
    


# In[117]:


get_recommendations('Avatar')


# Quality of recommender is not great, next step is to use metatdata to generate content based recommendations.

# Using top 3 actors, director, genre, keywords associated to the movie. Data is present in the form of "stringified" lists, we need to convert it to make it usable.

# In[118]:



# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']


# In[119]:


for feature in features: 
    df2[feature] = df2[feature].apply(literal_eval)


# In[120]:


# Get director's name from crew
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[121]:


# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []


# In[122]:


df2['director'] = df2['crew'].apply(get_director)


# In[123]:


features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)


# In[126]:


df2[['title_x', 'cast', 'director', 'keywords','genres']].head(3)


# In[132]:


# function to convert all strings to lower cases amd strip name spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ","")) for i in x]
    else:
        if isinstance(x, list):
            return str.lower(i.replace(" ",""))
        else:
            return ''


# In[133]:


# Apply clean data function to all features

features = ['cast', 'director', 'keywords','genres']

for feature in features:
    df2[feature].apply(clean_data)


# In[138]:


# Create metadata soup
def create_soup(x):
    return ' '.join(str(x['keywords']))+' '+' '.join(str(x['cast']))+' '+str(x['director'])+' '+' '.join(x['genres'])

df2['soup'] = df2.apply(create_soup, axis=1)


# In[140]:


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])


# In[141]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)


# In[143]:


df2 = df2.reset_index()


# In[147]:


indices = pd.Series(df2.index, index = df2['title_x'])


# In[149]:


# Reusing get_recommendations function by passing cosine_sim2 as the new argument

get_recommendations('The Dark Knight Rises', cosine_sim2)


# In[150]:


get_recommendations('The Godfather', cosine_sim2)


# ## Collaborative Filtering

# Previous method does not do a good job of predicting the taste of users and hence the recommendations are not personalized

# In[3]:


import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

reader = Reader()
ratings = pd.read_csv('ratings_small.csv')
ratings.head()


# In[4]:


data = Dataset.load_from_df(ratings[['userId','movieId','rating']], reader)


# In[5]:


data.split(n_folds = 5)


# In[6]:


algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[7]:


trainset = data.build_full_trainset()


# In[9]:


algo.fit(trainset)


# In[10]:


ratings[ratings['userId'] == 1]


# In[12]:


algo.predict(1, 303)


# In[ ]:




