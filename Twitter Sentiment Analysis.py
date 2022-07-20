#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from textblob import TextBlob


# In[2]:


import warnings as wg
wg.filterwarnings("ignore")


# In[3]:


get_ipython().run_line_magic('run', './key.ipynb')


# ### Connecting to jump server of twitter

# In[4]:


auth = tweepy.OAuthHandler(consumer_key, consumer_key_sec)


# ### Connecting jump server to web server of twitter

# In[5]:


auth.set_access_token(access_token,access_token_sec)


# ### Connecting to API Strong Server of Twitter

# In[6]:


api = tweepy.API(auth)


# In[7]:


keyword = input("Keyword you need analysis on : ")


# In[8]:


no_of_tweets = 1000
tweets = []
likes = []
time = []
for i in tweepy.Cursor( api.search_tweets,q=keyword, tweet_mode = "extended").items(no_of_tweets):
    tweets.append(i.full_text)
    likes.append(i.favorite_count)
    time.append(i.created_at)


# In[9]:


df=pd.DataFrame({'tweets': tweets, 'likes':likes,'time':time})
df


# ### Cleaning tweets

# In[10]:


def cleanTxt(text):
    text = re.sub(r"@[A-Za-z0-9]+" , '', text)        #remove @mentions
    text = re.sub(r'#', '', text)                     #removing "#" symbol
    text = re.sub(r'RT[\s]+', '', text)               # removing RT(retweets)
    text = re.sub(r'https?:\/\/\S+', '', text)        # remove the hyper link
    return text
df["tweets"]=df["tweets"].apply(cleanTxt)
df


# In[11]:


#Plot The Word Cloud
allWords = ' '.join([twts for twts in df['tweets']])
wordCloud = WordCloud(width=800,height=500,random_state=21,max_font_size=119).generate(allWords)
plt.imshow(wordCloud,interpolation="bilinear")
plt.axis('off')
plt.show()


# ### Creating a function to get subjectivity

# In[12]:


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity             


# ### Creating  a function to get the polarity

# In[13]:


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


# ##### Creating two new columns for getting subjectivity and polarity

# In[14]:


df['Subjectivity'] = df['tweets'].apply(getSubjectivity)
df['Polarity'] = df['tweets'].apply(getPolarity)
df


# ### Creating  a function to compute the positive , negative and neutral analysis

# In[15]:


def getAnalysis(score):
    if score>0:
        return 'Positive'
    elif score == 0 :
        return 'Neutral'
    else:
        return 'Negative'
df['Analysis']= df['Polarity'].apply(getAnalysis)
df


# ### Ploting the polarity and subjectivity

# In[17]:


plt.figure(figsize=(8,6))
for i in range (0,df.shape[0]):
    plt.scatter(df['Polarity'][i],df['Subjectivity'][i], color = 'Green')
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.grid()
plt.show()


# ### Showing the value count

# In[18]:


df['Analysis'].value_counts()


# ### Ploting and Visualizing the value count

# In[19]:


plt.title('Sentiment Analysis')
df['Analysis'].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.show()

