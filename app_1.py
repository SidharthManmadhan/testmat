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

goals = ['Start my Career with a Spring Week','Get a Summer Internship','Get an Internship alongside my Studies', 'Land a Placement Year','Win Awards & Competitions','Secure a Graduate Job','Find a Co-founder & Start a Business', 'Meet like-minded students','Expand my Network & Meet Industry Leader']
Goals =  st.multiselect('Enter the goals',goals,key = "one")

interest = st.multiselect('Enter the interest',df_tags['name'].unique(),key = "two")
weight = [1,2,3,3,2,1,1,2,3,3,2,1,1,2,3,3,2,1,1,2,3,3,2,1,1,2,3,3,2,1,1,2,3,3,2,1,1,2,3,3,2,1,1,2,3,3,2,1,1,2,3,3,2,1,1,2,3,3,2,1,1,2,3,3,2,1,3,3,2,1,1,2,3,3,2]
Weight = st.multiselect('Enter the weight',weight,key = "three")
Interest = pd.DataFrame(interest,columns = ['Interest'])
Weight = pd.DataFrame(Weight,columns = ['Weight'])
df_interest = pd.concat([Interest,Weight],axis = 1)
University = st.selectbox('Enter the university',df_universities['name'].unique(),key = 'four')
Degree =  st.selectbox('Enter the degree',df_degrees['name'].unique(),key = 'five')
Subject = st.selectbox('Enter the subject',df_subjects['subject_name'].unique(),key = 'six')
#Subject = pd.DataFrame(Subject,columns = ['Subject'])
#Score = [1]
#subjectscore = pd.DataFrame(Score,columns = ['subject score'])
#df_subject = pd.concat([Subject,subjectscore],axis = 1)
year = ['First Year ','Second Year','Third Year','Final Year']
Year = st.selectbox('Enter the year',year,key = 'seven')
data = []
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
   df_T =  pd.merge(df, df_interest, left_on='name',right_on='Interest',suffixes=('', '_x'),how = 'inner')
   df_T = df_T.loc[:,~df_T.columns.duplicated()]
   df_T['description'] = df_T['description'].str.replace(r'<[^<>]*>', '', regex=True)
   df_T['description'] = df_T['description'].str.replace(r'[^\w\s]+', '', regex=True)
   df_T['description'] = df_T['description'].str.lower()
   df_T['Interest'] = df_T['Interest'].str.lower()
   model = SentenceTransformer('bert-base-nli-mean-tokens')
   sentence_embeddings = model.encode(df_T['description'])
   word_embeddings = model.encode(df_T['Interest'])
   A = np.arange(len(df_T['description']))
   for a in A:
      description_score  =  distance.cdist([sentence_embeddings[a]],word_embeddings[0:])
      a += 1
      for x in description_score:
            df_T['description_score']=pd.Series(x)
     
   df_T['city score'] = np.nan
   df_universities = pd.merge(df_universities, df_cities, left_on='city_id',right_on='id',suffixes=('', '_x'),how = 'inner')
   df_universities = df_universities.loc[:,~df_universities.columns.duplicated()]
   df_universities = df_universities.loc[df_universities['name'] == University]
   city_name = df_universities.iloc[0]['city_name']
   df_T['city score'] = np.where(df_T['city_name'] == city_name, 1,0)
 #df_T = df_T[['id','touchpointable_id','kind', 'title','name','createable_for_name','city_name','Weight','description','description_score','city score']].copy()
   df_subjects =  df_subjects.loc[df_subjects['subject_name'] == Subject]
   subject_name = df_subjects.iloc[0]['subject_name']  
 
   S = []

     
   if ',' in subject_name:
        subject_0 = subject_name.split(', ')
        subject_0 = subject_0[0]
        S.append(subject_0)
  
  
   if '&' in subject_name:
        subject = subject_name.split('&')


        subject = subject[0]
        subject_1 = subject_name.split('&')[1]
        S.append(subject_1)
  
  

        if ',' in subject:
             subject_3 = subject.split(',')[1]
             S.append(subject_3)

    
        else:
            subject_3 = subject

            S.append(subject_3)


   else:
          S.append(subject_name)
   S = [x.strip(' ') for x in S]


   df_subject = pd.DataFrame(S, columns =['subject'])
   df_subject['subject_score'] = pd.Series([1 for x in range(len(df_subject.index))])
   df_S =  pd.merge(df, df_subject, left_on='name',right_on='subject',suffixes=('', '_x'),how = 'inner')
   df_S = df_S.loc[:,~df_S.columns.duplicated()]
   df_S = pd.merge(df_T, df_S, left_on='touchpointable_id',right_on='touchpointable_id',suffixes=('', '_x'),how = 'outer')
   df_S = df_S.loc[:,~df_S.columns.duplicated()]
   df_degrees = df_degrees.loc[df_degrees['name'] == Degree]
   degree = df_degrees.iloc[0]['name']
   degree = [degree]
   df_degree = pd.DataFrame(degree, columns =['degree'])
   df_degree['degree_score'] = pd.Series([1 for x in range(len(df_degree.index))])
   df_degree =  pd.merge(df, df_degree, left_on='name',right_on='degree',suffixes=('', '_x'),how = 'inner')
   df_degree = df_degree.loc[:,~df_degree.columns.duplicated()]
   O = ['Open to All Students']
   df_degree_1 = pd.DataFrame(O, columns =['degree'])
   df_degree_1['degree_score'] = pd.Series([0 for x in range(len(df_degree_1.index))])
   df_degree_1 =  pd.merge(df, df_degree_1, left_on='name',right_on='degree',suffixes=('', '_x'),how = 'inner')
   df_degree_1 = df_degree_1.loc[:,~df_degree_1.columns.duplicated()]
   df_degree_2 = pd.DataFrame(year, columns =['Year'])
   df_degree_2['year_score'] = pd.Series([1 for x in range(len(df_degree_2.index))])
   df_degree_2 =  pd.merge(df, df_degree_2, left_on='name',right_on='Year',suffixes=('', '_x'),how = 'inner')
   df_degree_2 = df_degree_2.loc[:,~df_degree_2.columns.duplicated()]
   df_D = pd.concat([df_degree,df_degree_1])
   df_D = pd.concat([df_D,df_degree_2])
    
   df_SD = pd.merge(df_S, df_D, left_on='touchpointable_id',right_on='touchpointable_id',suffixes=('', '_x'),how = 'inner')
   df_SD = df_SD.loc[:,~df_SD.columns.duplicated()]
   df_companies = pd.read_sql('select * from companies', con=engine)
   df_tagging = pd.read_sql('select * from taggings', con=engine)
   df_companies =  pd.merge(df_companies, df_tagging, left_on='id',right_on='taggable_id',suffixes=('', '_x'))
   df_tags = pd.read_sql('select * from tags', con=engine)
   df_companies = pd.merge(df_companies,df_tags,left_on='tag_id',right_on='id',suffixes=('', '_x'))
   df_companies = df_companies.loc[:,~df_companies.columns.duplicated()]
   df_companies['company score'] = pd.Series([1 for x in range(len(df_companies.index))])
   df = pd.merge(df,df_companies,left_on='name',right_on='name',suffixes=('', '_x'))
   df = df.loc[:,~df.columns.duplicated()]
   df = pd.merge(df_SD, df, left_on='touchpointable_id',right_on='touchpointable_id',suffixes=('', '_x'),how = 'outer')
   df = df.groupby('id', as_index=False).first()
   df = df[['id','touchpointable_id','kind', 'title','name','creatable_for_name','city_name','Weight','description','city score','subject_score','degree_score','company score']].copy()
   col_list = ['Weight','city score','degree_score','subject_score','company score']
   df['matching score'] = df[col_list].sum(axis=1)
   df = df.sort_values(by='matching score',ascending=False)
   st.write(df)
     
     

