import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Create and clean Election 2018 dataframe
elect18 = pd.read_csv('/home/xristsos/Documents/2016 election/national-files/us-senate-wide.csv')
elect18 = pd.DataFrame(elect18[['state','county','total.votes']])
elect18.columns = ['state','county','total_votes_senate18']
elect18.county = elect18.county + ' County'
elect18.sort_values('county',inplace=True)
elect18.set_index('county',inplace=True)
elect18.shape
elect18.head()
## Create and clean Election 2016 dataframe
elect16 = pd.read_json('/home/xristsos/Downloads/usa-2016-presidential-election-by-county.json')
elect16.shape
elect16.head()
### Bulk of the dataframe was embeded in a dictionary in the feilds column
df = pd.DataFrame(data=[x for x in elect16['fields']])
### Save a copy for geo info
geo_df = df['geo_shape'].copy()
sorted(list(df.index))

## Fill or drop null values
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
def remove_cols(df):
    for col in df.columns:
        if '16' in col:
            df.drop(col,axis=1,inplace=True)
        elif df[col].isna().sum() > 40:
            df.drop(col,axis=1,inplace=True)
remove_cols(df)
df.voter_percent.fillna(df.voter_percent.mean(),inplace=True)
df.non_voters_percent.fillna(df.non_voters_percent.mean(),inplace=True)
df.tail()
## Join new data frames
df = pd.merge(elect18,df,how='inner',on=['county','state'])

df.drop_duplicates(inplace=True)
df['2018_vote_percent'] = df.total_votes_senate18 / df.total_population
df.head()

### Future geo cleaning
geo = pd.read_json('/home/xristsos/flatiron/projects/where-are-missing-voters/data/gz_2010_us_050_00_20m.json')
geo.head()
