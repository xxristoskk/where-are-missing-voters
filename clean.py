import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# pop = pd.read_json('/home/xristsos/Downloads/us-population-urban-area.json')
elect = pd.read_json('/home/xristsos/Downloads/usa-2016-presidential-election-by-county.json')

##Election 2016 dataset
# df.columns
df = pd.DataFrame(data=[x for x in elect['fields']])

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
##create a new catagorical column 'winner'
def define_winner(df):
    c_t_wins =[]
    for x in range(len(df)):
        if df.votes16_trumpd[x] > df.votes16_clintonh[x]:
            c_t_wins.append('Trump')
        else:
            c_t_wins.append('Clinton')
    return c_t_wins
df['winner'] = define_winner(df)
# df = pd.DataFrame(df[df.voter_percent > 0.25])
##create a new column 'non_voters' which takes the difference of the total population and number of votes
df['non_voters'] = df['total_population'] - df['votes']
df['non_voters_percent'] = df.non_voters / df.total_population
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
df.voter_percent.isna().sum()
sns.heatmap(df.isna())
df.non_voters_percent = (df.non_voters_percent - df.non_voters_percent.mean()) / df.non_voters_percent.std()
df.voter_percent = (df.non_voters_percent - df.non_voters_percent.mean()) / df.non_voters_percent.std()
# df.dropna(inplace=True)
##create new dataframes based on size of county
# df1 = pd.DataFrame(df[df['size_rank']==1])
# df2 = pd.DataFrame(df[df['size_rank']==2])
df = pd.DataFrame(df[df['size_rank']==3])
# df4 = pd.DataFrame(df[df['size_rank']==4])
# df5 = pd.DataFrame(df[df['size_rank']==5])

def create_dict(df,col):
    """creates a list of df's"""
    unique_col = df[col].unique()
    df_list = []
    for val in unique_col:
        df_list.append(df[df[col]==val])
    return df_list
all_states = create_dict(df,'state')

def run_all_model(all):
    models = {}
    target = 'voter_percent'
    x_cols = ['teen_births','less_than_high_school','poverty_rate_below_federal_poverty_threshold',
              'children_in_single_parent_households','gini_coefficient','median_earnings_2010_dollars']
    for d in all:
        state = list(d['state'])[0]
        d = d[['state','voter_percent','teen_births','less_than_high_school','median_earnings_2010_dollars','poverty_rate_below_federal_poverty_threshold',
               'at_least_bachelor_s_degree','gini_coefficient','children_in_single_parent_households']]
        d = d[d['voter_percent']>-1.5]
        d = d[d['voter_percent']<2]
        for col in x_cols:
            d[col] = (d[col] - d[col].mean())/d[col].std()
        predictors = '+'.join(x_cols)
        formula = target + "~" + predictors
        try:
            model = ols(formula=formula, data=d).fit()
        except:
            print('Tuple not in range or something')
            model=0
        models[state] = model
    return models

all_models = run_all_model(all_states)

for key, val in all_models.items():
    if val == 0:
        continue
    else:
        print(f'State: {key}')
        print(f'P-values: \n {val.pvalues}')
        print(f'R-squared: \n {val.rsquared}')

### DECISION TREE ATTEMPT FAILED
# all_states.keys()
# i = all_states['Colorado']
# sns.set(style='ticks')
# fig = plt.figure(figsize=(15,15))
# sns.scatterplot(x=df['voter_percent'],y=df['management_professional_and_related_occupations'],hue=df['winner'])


#break the colums in to groups to plot 4 on a row at a time
# df['norm_hs'] = (df.less_than_high_school - df.less_than_high_school.mean()) / df.less_than_high_school.std()
# df.norm_hs.plot.hist()
# features = ['teen_births','less_than_high_school','poverty_rate_below_federal_poverty_threshold','construction_extraction_maintenance_and_repair_occupations',
#             'production_transportation_and_material_moving_occupations','uninsured','children_in_single_parent_households',
#             'child_poverty_living_in_families_below_the_poverty_line','adult_obesity','sexually_transmitted_infections']
# n = 4
# row_groups= [features[i:i+n] for i in range(0, len(features), n)]
# for i in row_groups:
#     pp = sns.pairplot(data=df,y_vars='voter_percent',x_vars=i, kind="reg", height=5)
# for i in row_groups:
#     pp = sns.pairplot(data=df1,y_vars='voter_percent',x_vars=i, kind="reg", height=5)
# for i in row_groups:
#     pp = sns.pairplot(data=df2,y_vars='voter_percent',x_vars=i, kind="reg", height=5)
# for i in row_groups:
#     pp = sns.pairplot(data=df,y_vars='voter_percent',x_vars=i, kind="reg", height=5)
# for i in row_groups:
#     pp = sns.pairplot(data=df4,y_vars='voter_percent',x_vars=i, kind="reg", height=5)
# for i in row_groups:
#     pp = sns.pairplot(data=df5,y_vars='voter_percent',x_vars=i, kind="reg", height=5)

# df.plot.scatter(x='teen_births',y='voter_percent')
# df.plot.scatter(x='less_than_high_school',y='voter_percent')
# df.plot.scatter(x='poverty_rate_below_federal_poverty_threshold',y='voter_percent')
# sns.pairplot(df,x_vars=X_train,y_vars=y_train,kind='scatter',height=3)
# df['voter_percent']
# df['poverty_rate_below_federal_poverty_threshold']



from sklearn.model_selection import train_test_split
y = df['voter_percent']
X = df
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42)

m = model = ols(formula=formula,data=X_train).fit()
m.summary()


## corr heatmap
corr = df.corr()
def CorrMtx(df, dropDuplicates = True):

    # Your dataset is already a correlation matrix.
    # If you have a dateset where you need to include the calculation
    # of a correlation matrix, just uncomment the line below:
    # df = df.corr()

    # Exclude duplicate correlations by masking uper right values
    if dropDuplicates:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Set background color / chart style
    sns.set_style(style = 'white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw correlation plot with or without duplicates
    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap,
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    else:
        sns.heatmap(df, cmap=cmap,
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)

CorrMtx(corr, dropDuplicates = True)
# df['less_than_high_school'] = np.log(df['less_than_high_school'])
subset.voter_percent.describe()
sns.jointplot(x=df.poverty_rate_below_federal_poverty_threshold,y=df.voter_percent,data=subset,kind='reg')
# sns.jointplot(x=df.norm_hs,y=df.norm_percent,data=df,kind='reg')



df = pd.DataFrame(df[df['size_rank']==3])
df = subset
target = 'voter_percent'
x_cols = ['teen_births','less_than_high_school','poverty_rate_below_federal_poverty_threshold',
          'children_in_single_parent_households','gini_coefficient','median_earnings_2010_dollars']
for col in x_cols:
    df[col] = (df[col] - df[col].mean())/df[col].std()
# df.head()
from statsmodels.formula.api import ols
predictors = '+'.join(x_cols)
formula = target + '~' + predictors
model = ols(formula=formula,data=df).fit()
model.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[x_cols]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
list(zip(x_cols, vif))


import scipy.stats as stats
fig = sm.graphics.qqplot(model.resid, dist=stats.norm, line='45', fit=True)
plt.scatter(model.predict(subset[x_cols]), model.resid)
plt.plot(model.predict(subset[x_cols]), [0 for i in range(len(subset))])

for i in range(0,10):
    q = i / 100
    print('{} percentile: {}'.format(q, df['voter_percent'].quantile(q=q)))
subset = df[df['voter_percent']>-1.5]
subset = subset[subset['voter_percent']<2]
# subset = df[df['voter_percent']>.2]
print('Percent removed:',(len(df) - len(subset))/len(df))

predictors = '+'.join(x_cols)
formula = target + "~" + predictors
model = ols(formula=formula, data=subset).fit()
model.summary()
