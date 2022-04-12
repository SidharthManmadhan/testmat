import streamlit as st
import pandas as pd
import psycopg2 as pg
import collections, functools, operator
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial import distance

engine = pg.connect("dbname='huzzle_staging' user='postgres' host='huzzle-staging.ct4mk1ahmp9p.eu-central-1.rds.amazonaws.com' port='5432' password='2Yw*PG9x-FcWvc7R'")
df_tags = pd.read_sql('select * from tags', con=engine)
df_degrees = pd.read_sql('select * from degrees', con=engine)
df_degrees['name'] = df_degrees['name'].replace(["Bachelor's"],['Bachelors'])
df_degrees['name'] = df_degrees['name'].replace(["Master's"],['Masters'])
df_universities = pd.read_sql('select * from universities', con=engine)
df_subjects = pd.read_sql('select * from subjects', con=engine)
df_subjects.rename(columns = {'name':'subject_name'}, inplace = True)
goal_0 = [{'Spring Weeks':7, 'Virtual Internship':3, 'Career Fairs':3,'Insight Days':5,'Competitions':2}]                                      #Initialing touchpoint weights,later on this will be converted to dataframe
goal_1 = [{'Summer Internship':10, 'Networking & Social':3,'Career Fairs':3,'Insight Days':2,'Workshops':2}]
goal_2 = [{'Virtual Internship':3, 'Off-cycle':7, 'Networking & Social':5,'Career Fairs':2,'Conferences & Talks':3}]
goal_3 = [{'Placement Programme':10, 'Networking & Social':2,'Career Fairs':3,'Insight Days':3,'Workshops':2}]
goal_4 = [{'Conferences & Talks':2,'Workshops':2,'Competitions':6}]
goal_5 = [{'Networking & Social':3,'Career Fairs':7,'Jobs':10}] 
goal_6 = [{'Networking & Social':4,'Conferences & Talks':3,'Competitions':3}]
goal_7 = [{'Networking & Social':3,'Workshops':2,'Conferences & Talks':2,'Competitions':3}]
goal_8 = [{'Networking & Social':2,'Career Fairs':2,'Insight Days':2,'Workshops':2,'Conferences & Talks':2}]
goal_9 = [{'no goals'}]

goal_dataframe_mapping = {
    'Start my Career with a Spring Week':goal_0,                                   #mapping  goals with touchpoints
    'Get a Summer Internship':goal_1,
    'Get an Internship alongside my Studies':goal_2,
    'Land a Placement Year':goal_3,
    'Win Awards & Competitions':goal_4,
    'Secure a Graduate Job':goal_5,
    'Find a Co-founder & Start a Business':goal_6,
    'Meet Like-minded Students & join Societies':goal_7,
    'Expand my Network & Connect with Industry Leaders':goal_8,
    'No goals selected' : goal_9} 
data = []
goals = ['Start my Career with a Spring Week','Get a Summer Internship','Get an Internship alongside my Studies', 'Land a Placement Year','Win Awards & Competitions','Secure a Graduate Job','Find a Co-founder & Start a Business', 'Meet like-minded students','Expand my Network & Meet Industry Leader']

Goals =  st.multiselect('Enter the goals',goals,key = "one")
interest = st.multiselect('Enter the interest',df_tags['name'].unique(),key = "two")
weight = [1,2,3,3,2,1]
Weight = st.multiselect('Enter the weight',weight,key = "three")
Interest = pd.DataFrame(interest,columns = ['Interest'])
Weight = pd.DataFrame(Weight,columns = ['Weight'])
df_interest = pd.concat([Interest,Weight],axis = 1)
University = st.selectbox('Enter the university',df_universities['name'].unique(),key = 'four')
Degree =  st.selectbox('Enter the degree',df_degrees['name'].unique(),key = 'five')
Subject = st.selectbox('Enter the subject',df_subjects['subject_name'].unique(),key = 'six')
year = [1,2,3,4]
Year = st.selectbox('Enter the year',year,key = 'seven')
for x in Goals:
     data.append(pd.DataFrame(goal_dataframe_mapping[x])) #based on the goals selected corresponding dataframes are printed
     result = dict(functools.reduce(operator.add,map(collections.Counter, data)))   #if same touchpoints are available on goals selected, the values of the touchpoints are added to each other and list will be formed 
#result = {i:round(j/user_input) if j>1 else j for i,j in result.items()} 
#result = {i:round(j/user_input) for i,j in result.items()}
     df_goals =  pd.DataFrame(result.items(), columns=['kind_1', 'value'])  
    
     
     df_touchpoints = pd.read_sql('select * from touchpoints', con=engine)
     grouped_1 = df_touchpoints.groupby(df_touchpoints.state)
     df_touchpoints = grouped_1.get_group(1)
     grouped_2 = df_touchpoints.groupby(df_touchpoints.touchpointable_type)
     df_jobs = grouped_2.get_group("Job")
     df_1 = pd.read_sql('select * from jobs', con=engine)
     df_jobs =  pd.merge(df_jobs, df_1, left_on='touchpointable_id',right_on='id',suffixes=('', '_x'))
     df_jobs = df_jobs.loc[:,~df_jobs.columns.duplicated()]
     df_tagging = pd.read_sql('select * from taggings', con=engine)
     df_jobs =  pd.merge(df_jobs, df_tagging, left_on='id',right_on='taggable_id',suffixes=('', '_x'))
     df_tags = pd.read_sql('select * from tags', con=engine)
     df_jobs = pd.merge(df_jobs,df_tags,left_on='tag_id',right_on='id',suffixes=('', '_x'))
     df_jobs = df_jobs.loc[:,~df_jobs.columns.duplicated()]
     df_jobs['kind'] = df_jobs['kind'].replace([0,1],['Jobs','Jobs'])

     df_jobs['new_col'] = range(1, len(df_jobs) + 1)
     df_jobs = df_jobs.set_index('new_col')

     grouped_3 = df_touchpoints.groupby(df_touchpoints.touchpointable_type)
     df_events = grouped_3.get_group("Event")
     df_2 = pd.read_sql('select * from events', con=engine)
     df_events =  pd.merge(df_events, df_2, left_on='touchpointable_id',right_on='id',suffixes=('', '_x'))
     df_tagging = pd.read_sql('select * from taggings', con=engine)
     df_events =  pd.merge(df_events, df_tagging, left_on='id',right_on='taggable_id',suffixes=('', '_x'))
     df_tags = pd.read_sql('select * from tags', con=engine)
     df_events = pd.merge(df_events,df_tags,left_on='tag_id',right_on='id',suffixes=('', '_x'))
     df_events = df_events.loc[:,~df_events.columns.duplicated()]
#df_events = df_events.loc[df_events["kind"] != 0]
     df_events['kind'] = df_events['kind'].replace([0,1,2,3,4,5,6,7],['Networking & Social','Networking & Social','Career Fairs','Insight Days','Workshops','Conferences & Talks','Conferences & Talks','Competitions'])
     df_events['new_col'] = range(1, len(df_events) + 1)
     df_events = df_events.set_index('new_col')

     grouped_4 = df_touchpoints.groupby(df_touchpoints.touchpointable_type)
     df_internship = grouped_4.get_group("Internship")
     df_3 = pd.read_sql('select * from internships', con=engine)
     df_internship =  pd.merge(df_internship, df_3, left_on='touchpointable_id',right_on='id',suffixes=('', '_x'))
     df_tagging = pd.read_sql('select * from taggings', con=engine)
     df_internship =  pd.merge(df_internship, df_tagging, left_on='id',right_on='taggable_id',suffixes=('', '_x'))
     df_tags = pd.read_sql('select * from tags', con=engine)
     df_internship = pd.merge(df_internship,df_tags,left_on='tag_id',right_on='id',suffixes=('', '_x'))
     df_internship = df_internship.loc[:,~df_internship.columns.duplicated()]
     df_internship['kind'] = df_internship['kind'].replace([0,1,2,3,4],['Spring Weeks','Summer Internship','Off-cycle','Winter','Virtual Internship'])
     df_internship['new_col'] = range(1, len(df_internship) + 1)
     df_internship = df_internship.set_index('new_col')

     df_4 = pd.concat([df_jobs,df_events])
     df = pd.concat([df_4,df_internship])
     df_tc = pd.read_sql('select * from touchpoints_cities', con=engine)
     df = pd.merge(df,df_tc,left_on='id',right_on='touchpoint_id',suffixes=('', '_x'),how = 'left')
     df = df.loc[:,~df.columns.duplicated()]
     df_cities = pd.read_sql('select * from cities', con=engine)
     df_cities.rename(columns = {'name':'city_name'}, inplace = True)
     df = pd.merge(df,df_cities,left_on='city_id',right_on='id',suffixes=('', '_x'),how = 'left')
     df = df.loc[:,~df.columns.duplicated()]
     df =  pd.merge(df, df_goals, left_on='kind',right_on='kind_1',suffixes=('', '_x'),how = 'inner')
     df = df.loc[:,~df.columns.duplicated()]
     #grouped_5 = df.groupby(df.type)
     #df_T = grouped_5.get_group("Topic")
     df =  pd.merge(df, df_interest, left_on='name',right_on='Interest',suffixes=('', '_x'),how = 'left')
     df_T = df.loc[:,~df.columns.duplicated()]
     df['description'] = df['description'].str.replace(r'<[^<>]*>', '', regex=True)
     df['description'] = df['description'].str.replace(r'[^\w\s]+', '', regex=True)
     df['description'] = df['description'].str.lower()
     df['name'] = df['name'].str.lower()
     model = SentenceTransformer('bert-base-nli-mean-tokens')
     sentence_embeddings = model.encode(df['description'])
     word_embeddings = model.encode(df['name'])
     A = np.arange(len(df['description']))
     for a in A:
          description_score  =  distance.cdist([sentence_embeddings[a]],word_embeddings[0:])
          a += 1
          for x in description_score:
               df['description_score']=pd.Series(x)
     
     df['city score'] = np.nan
     df_universities = pd.merge(df_universities, df_cities, left_on='city_id',right_on='id',suffixes=('', '_x'),how = 'inner')
     df_universities = df_universities.loc[:,~df_universities.columns.duplicated()]
     df_universities = df_universities.loc[df_universities['name'] == University]
     city_name = df_universities.iloc[0]['city_name']
     df['city score'] = np.where(df['city_name'] == city_name, 1,0)
     #grouped_7 = df.groupby(df.type)
     #df_E = grouped_7.get_group("EducationRequirement")
     df['degree score'] = np.nan
     df_degrees = df_degrees.loc[df_degrees['name'] == Degree]
     degree = df_degrees.iloc[0]['name']
     df['degree score'] = np.where(df['name'] == degree ,1,0)
     df['subject score'] = np.nan
     df_subjects =  df_subjects.loc[df_subjects['subject_name'] == Subject]
     subject_name = df_subjects.iloc[0]['subject_name']
     df['subject score'] = np.where(df['name'] == subject_name ,1,0)
     col_list = ['Weight','description_score','city score','degree score','subject score']
     df['matching score'] = df[col_list].sum(axis = 1)
     
     
     df = df.groupby('id', as_index=False).first()


     




     

     #grouped_8 = df.groupby(df.name)
     #df_O =  grouped_8.get_group("Open to All Students")
     #df = pd.merge(df_TA,df_O,left_on='id',right_on='id',suffixes=('', '_x'),how = 'inner')
     #df = df.loc[:,~df.columns.duplicated()]
     #df = pd.concat([df_TAE,df])





     #df['subject_score'] = np.nan
     #df['subject_score'] = np.nan
     #df_subjects =  df_subjects.loc[df_subjects['subject_name'] == Subject]
     #subject_name = df_subjects.iloc[0]['subject_name']
    
     
     #columns_list = ['Weight','matching score', 'degree score','subject_score_0','subject_score_1']

     #df['matching score'] = df[columns_list].sum(axis = 1)
     #df['description'] = df['description'].str.replace(r'<[^<>]*>', '', regex=True)
     #df['description'] = df['description'].str.replace(r'[^\w\s]+', '', regex=True)
     #df['description'] = df['description'].str.lower()
     #df['name'] = df['name'].str.lower()
     #model = SentenceTransformer('bert-base-nli-mean-tokens')
     #sentence_embeddings = model.encode(df['description'])
     #word_embeddings = model.encode(df['name'])
     #A = np.arange(len(df['description']))
     #for a in A:
          #description_score  =  distance.cdist([sentence_embeddings[a]],word_embeddings[0:])
          #a += 1
          #for x in description_score:
               #df['description_score']=pd.Series(x)





     

     #df.loc[df['city_name'] == city_name, 'matching score'] = 1

     #df['description'] = df['description'].str.replace(r'<[^<>]*>', '', regex=True)
     #df['description'] = df['description'].str.replace(r'[^\w\s]+', '', regex=True)
     #df['description'] = df['description'].str.lower()
     #df['name'] = df['name'].str.lower()
     #model = SentenceTransformer('bert-base-nli-mean-tokens')
     #sentence_embeddings = model.encode(df['description'])
     #word_embeddings = model.encode(df['name'])  
     #A = np.arange(len(df['description']))
    # for a in A:
         #matching_score  =  cosine_similarity([word_embeddings[a]],sentence_embeddings[0:])
         #break
         #a += 1
         #for x in matching_score:
                #df['matching_score']=pd.Series(x)

     
     #df_universities = pd.merge(df_universities, df_cities, left_on='city_id',right_on='id',suffixes=('', '_x'),how = 'inner')
     #df_universities = df_universities.loc[:,~df_universities.columns.duplicated()]
     #df_universities = df_universities.loc[df_universities['name'] == University]
     #df = pd.merge(df, df_universities, left_on='city_name',right_on='city_name',suffixes=('', '_x'),how = 'outer')
     #df = df.loc[:,~df.columns.duplicated()]

     #df_subject = df_subjects.loc[df_subjects['subject_name'] == Subject]
     #df = pd.merge(df, df_subject, left_on='name',right_on='subject_name',suffixes=('', '_x'),how = 'outer')
     #df = df.loc[:,~df.columns.duplicated()]
     #df = df.groupby('id').agg(lambda x: x.tolist())




     st.write(df)

     
     



     
