# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 21:09:00 2019

@author: Pratik
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PopularSongs_Recomm():
    
    def __init__(self) :
        self.train = None
        self.User_Id =None
        self.coocurrence_matrix = None
    def Popularity_level(self,train,User_Id):
        self.train=train
        self.User_Id=User_Id
        #adding Column in dataframe based on Listen_Count Conditions 
        df=train
        lc=df['Listen_Count']
        
        lst1=[]
        for x in lc:
            if x >=250:
                lst1.append('High')
            elif x >= 100 and x < 250:
                lst1.append('Medium')
            else :
                lst1.append('Low')
        df['Popularity']=lst1 
        df['User_Id']=User_Id
        high_pop=df[df['Popularity']=='High']
        samp_high=high_pop.sample(7)
        med_pop=df[df['Popularity']=='Medium']
        samp_med=med_pop.sample(5)
        low_pop=df[df['Popularity']=='Low']
        
        samp_low=low_pop.sample(3)
        a=samp_high.append(samp_med,ignore_index=True)
        b = a.append(samp_low,ignore_index=True)
        result = b.sort_values(['Rank'],ascending=True)
        return result[['User_Id','Title','Rank','Popularity']]
       
class Item_Similarity_Recomm():
    def __init__(self):
        self.df=None
        self.userid=None
        self.song=None
        self.songlist=None
        
    def create(self,df,userid,song):
        self.df = df
        self.userid = userid
        self.song = song
        
    def get_User_Songs(self,userid):
        #users=self.df['User_Id'].unique()
        user_data=self.df[self.df['User_Id']==userid]
        songlist=list(user_data['Title'].unique())
        user_songs=list(dict.fromkeys(songlist))
        return user_songs
    
    def get_Song_Users(self,song):
        song_data =self.df[self.df['Title']==song]
        userslist = list(song_data['User_Id'].unique())
        song_users=list(dict.fromkeys(userslist))
        return song_users
        
    
    def get_All_Songs(self):
        self.all_songs=list(self.df['Title'].unique())
        return self.all_songs
    
    def generate_cooccurence_matrix(self,user_songs,all_songs):
        #get all users which listen to the same song present in our specific users song list
        our_users_song_users=[]
        for i in range(0,len(user_songs)):
            our_users_song_users.append(self.get_Song_Users(user_songs[i]))
        
        self.cooccurence_matrix= np.matrix(np.zeros(shape=(len(user_songs),len(all_songs))),float)
        #calculating similarity between our users song and all songs
        for i in range(0,len(all_songs)):
            #finding unique users of each song in all songs
            songs_i_data = self.df[self.df['Title'] == all_songs[i]]
            users_i = set(songs_i_data['User_Id'].unique())
            
            for j in range(0,len(user_songs)):
                #Get unique listeners (users) of song (item) j
                users_j = our_users_song_users[j]
                    
                #Calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    #Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    self.cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    self.cooccurence_matrix[j,i] = 0
                    
        
        return self.cooccurence_matrix
    
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value
        #Also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        columns = ['User_id', 'Title', 'Score', 'Rank']
        #index = np.arange(1) # array of numbers for the number of samples
        df = pd.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= 10:
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
        
    def recommend(self, user):
        #1. Get all unique songs for specific user
        user_songs = self.get_User_Songs(user)    
        print("No. of unique songs for the user: %d" % len(user_songs))
        #2. Get all unique items (songs) in the training data
        all_songs = self.get_All_Songs()
        print("no. of unique songs in the training set: %d" % len(all_songs))
        #3. Construct item cooccurence matrix of size len(user_songs) X len(songs)
        cooccurence_matrix = self.generate_cooccurence_matrix(user_songs, all_songs)
        #4. Use the cooccurence matrix to make recommendations
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations
    
    def get_similar_items(self, songlist):
        user_songs = songlist
        #Get all unique items (songs) in the training data
        
        all_songs = self.get_All_Songs()
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        #Construct item cooccurence matrix of size len(user_songs) X len(songs)
        
        cooccurence_matrix = self.generate_cooccurence_matrix(user_songs, all_songs)
        #Use the cooccurence matrix to make recommendations
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
         
        return df_recommendations
    
    
    
    
#if __name__=="__main__":
#    train=pd.read_csv(r"C:\Users\dbda\Desktop\Project\Codes\train.csv")
#    df=pd.read_csv(r"C:\Users\dbda\Desktop\Project\Codes\song_df.csv")
#    import sys
#    choice=0
#    while choice !=4:
#        print("1.Popularity Based Song Recommendation")
#        print("2.Personalized Song Recommendation")
#        print("3.Exit")
#        choice=int(input("Enter Your choice: "))
#        
#        if choice==1:
#            
#            #train=pd.read_csv(r"C:\Users\dbda\Desktop\Project\Codes\train.csv")
#            mod = PopularSongs_Recomm()
#            #print(ans)
#            user1='e006b1a48f466bf59feefed32bec6494495a4436'
#            res=mod.Popularity_level(train,user1)
#            print(res)
#        elif choice==2:
#            
#            song1='Missing You'
#            #df=pd.read_csv(r"C:\Users\dbda\Desktop\Project\Codes\song_df.csv")
#            
#            train_data, test_data = train_test_split(df, test_size = 0.20, random_state=0)
#            
#            #recommendations for First Person
#            mod1=Item_Similarity_Recomm()
#            mod1.create(train_data,user1,song1)
#            songs_list1=mod1.get_User_Songs(user1)
#            users_list1=mod1.get_Song_Users(song1)
#            all_songs1=mod1.get_All_Songs()
#            coo=mod1.generate_cooccurence_matrix(songs_list1,all_songs1)
#            p_recc1=mod1.recommend(user1)
#            print("Recommendation for first Person:")
#            print(p_recc1)
#            #Recommendations for second Person
#            mod2=Item_Similarity_Recomm()
#            user2='d6589314c0a9bcbca4fee0c93b14bc402363afea'
#            song2='Champion'
#            mod2.create(train_data,user2,song2)
#            songs_list2=mod1.get_User_Songs(user2)
#            users_list2=mod1.get_Song_Users(song2)
#            all_songs2=mod1.get_All_Songs()
#            coo=mod1.generate_cooccurence_matrix(songs_list2,all_songs2)
#            p_recc2=mod1.recommend(user2)
#            print("Recommendation for first Person:")
#            print(p_recc2)
#            mod2.get_similar_items(['God Is A DJ'])
#        elif choice==3:
#            sys.exit()