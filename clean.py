import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

## Income data I want to add to the main data frame
## Needs to be cleaned up more in order to merge
income_df = pd.read_excel('/home/xristsos/Documents/2016 election/lapi1118_0.xlsx')
income_df.head(10)
income_df.columns
income_df = pd.DataFrame(income_df.drop(income_df.index[[0,1,2,3,4]]))
income_df.columns = ['county','2015_dollars','2016_dollars','2017 dollars','rank_in_state','percent_change_2016','percent_change_2017','new_rank']

## List of states for a future function that will remove them from the data frame
states = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado','Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho','Illinois',
'Indiana','Iowa','Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts','Michigan','Minnesota','Mississippi','Missouri','Montana',
'Nebraska','Nevada','New Jersey','New Mexico','New York','North Carolina','North Dakota','Ohio','Oklahoma','Oregon','Pennsylvania','Rhode Island',
'South Carolina','South Dakota','Tennessee','Texas','Utah','Vermont','Virginia','Washington','West Virginia','Wisconsin','Wyoming']

income_df.head()
income_df.reset_index(inplace=True)
income_df.drop('index',axis=1,inplace=True)
income_df.sort_values(by='county',inplace=True)
income_df.shape
income_df.dropna(inplace=True)
income_df.head()
income_df.set_index('county',inplace=True)



## Election 2018 votes
elect18 = pd.read_csv('/home/xristsos/Documents/2016 election/national-files/us-senate-wide.csv')
elect18 = pd.DataFrame(elect18[['state','county','total.votes']])
elect18.columns = ['state','county','total_votes_senate18']
elect18.county = elect18.county + ' County'
elect18.sort_values('county',inplace=True)
elect18.set_index('county',inplace=True)
elect18.shape

##Election 2016 dataset
elect16 = pd.read_json('/home/xristsos/Downloads/usa-2016-presidential-election-by-county.json')

elect16.shape
## Fields column holds dictionaries for each county
df = pd.DataFrame(data=[x for x in elect16['fields']])
df.drop(['state','geo_shape','temp_bins'],axis=1,inplace=True)
df.rename(columns={'st':'state'},inplace=True)
df.county = [x.split(', ') for x in df.county]
df.county = [x[0] for x in df.county]
df.set_index('county',inplace=True)

##create a columnn for the percentage of voter turnout
df['voter_percent'] = df['votes'] / df['total_population']

## ranks the size of county from 1 to 5
scores = []
for i in df.total_population:
    if i <= 5000:
        scores.append(1)
    elif i > 5000 and i <= 15000:
        scores.append(2)
    elif i > 15000 and i <= 30000:
        scores.append(3)
    elif i > 30000 and i <= 50000:
        scores.append(4)
    elif i > 50000:
        scores.append(5)
df['size_rank'] = scores

##create a new column 'non_voters' which takes the difference of the total population and number of votes
df['non_voters'] = df['total_population'] - df['votes']

##Create a new columns for percent of non-voters
df['non_voters_percent'] = df.non_voters / df.total_population

##Fill and remove missing data
df.teen_births.fillna(df.teen_births.median(),inplace=True)
# def remove_cols(df):
#     for col in df.columns:
#         if '16' in col:
#             df.drop(col,axis=1,inplace=True)
#         elif df[col].isna().sum() > 40:
#             df.drop(col,axis=1,inplace=True)
# remove_cols(df)
df.voter_percent.fillna(df.voter_percent.mean(),inplace=True)
df.non_voters_percent.fillna(df.non_voters_percent.mean(),inplace=True)

## Join new data frames
all_df = pd.merge(elect18,df,how='right',on=['county','state'])
all_df.shape
all_df.isna().sum()
all_df.head()
all_df['2018_vote_percent'] = all_df.total_votes_senate18 / all_df.total_population
