# -*- coding: utf-8 -*-
"""
@author: Sahil Manchanda
"""
import numpy as np 
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input
from keras.utils import to_categorical
import pandas as pd

video_games_data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv', encoding = "ISO-8859-1")
video_games_data = video_games_data.dropna(axis=0)
video_games_data = video_games_data.loc[video_games_data["NA_Sales"]>1]
video_games_data = video_games_data.loc[video_games_data["EU_Sales"]>1]

Genre = video_games_data["Genre"].values
Publisher = video_games_data["Publisher"].values
platform = video_games_data["Platform"].values                
Rating = video_games_data["Rating"].values

Genre = pd.get_dummies(Genre)                          
Publisher = pd.get_dummies(Publisher)  
Platform = pd.get_dummies(platform)  
Rating = pd.get_dummies(Rating)  

X = video_games_data[["Critic_Score","Critic_Count","User_Score","User_Count"]].values
X = np.concatenate((X,Genre),axis = 1)
X = np.concatenate((X,Publisher),axis = 1)
X = np.concatenate((X,Platform),axis = 1)
X = np.concatenate((X,Rating),axis = 1)
                 

Na = video_games_data["NA_Sales"]
Eu = video_games_data["EU_Sales"]
JP = video_games_data["JP_Sales"]

a = Input(shape=(60,))
b = Dense(32,activation = 'relu')(a)
c = Dense(32,activation = 'relu')(b)
fin1 = Dense(1,activation = 'linear')(c)
fin2 = Dense(1,activation = 'linear')(c)
fin3 = Dense(1,activation = 'linear')(c)

model = Model(input = a,output=[fin1,fin2,fin3])
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(X,[Na, Eu, JP],nb_epoch=3000,batch_size=100, verbose=1)

