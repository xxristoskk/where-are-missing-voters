import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Reshape the main data frame  and add new/remove features
big_df = pd.DataFrame(all_df.drop(['voter_percent','2018_vote_percent'],axis=1))
big_df['avg_percent'] = (all_df['voter_percent'] + all_df['2018_vote_percent']) / 2
big_df['low_turnout'] = [1 if x <= .35 else 0 for x in big_df['avg_percent']]
big_df = pd.DataFrame(big_df[big_df['avg_percent']< .9])
big_df.low_turnout.sum()
big_df = pd.DataFrame(big_df[['gini_coefficient','low_turnout','size_rank','asian_american_population','african_american_population','at_least_high_school_diploma',
                      'child_poverty_living_in_families_below_the_poverty_line','uninsured','production_transportation_and_material_moving_occupations',
                      'poverty_rate_below_federal_poverty_threshold','construction_extraction_maintenance_and_repair_occupations',
                      'less_than_high_school','at_least_bachelor_s_degree','adults_65_and_older_living_in_poverty','unemployment','graduate_degree',
                      'latino_population','management_professional_and_related_occupations','sire_homogeneity','native_american_population',
                      'sales_and_office_occupations','white_not_latino_population']])
len(big_df.columns)

## plot eda
features = ['gini_coefficient','avg_percent','low_turnout','size_rank','asian_american_population','african_american_population','at_least_high_school_diploma',
                      'child_poverty_living_in_families_below_the_poverty_line','uninsured','production_transportation_and_material_moving_occupations',
                      'poverty_rate_below_federal_poverty_threshold','construction_extraction_maintenance_and_repair_occupations','total_population',
                      'less_than_high_school','at_least_bachelor_s_degree','adults_65_and_older_living_in_poverty','unemployment','graduate_degree',
                      'latino_population','management_professional_and_related_occupations','sire_homogeneity','native_american_population',
                      'sales_and_office_occupations','white_not_latino_population']
len(features)

n = 2
row_groups= [features[i:i+n] for i in range(0, len(features), n)]
for i in row_groups:
    pp = sns.pairplot(data=big_df,y_vars='avg_percent',x_vars=i, kind="reg", height=3)


## corr heatmap
corr = big_df[['gini_coefficient','low_turnout','size_rank','asian_american_population','african_american_population','at_least_high_school_diploma',
                      'child_poverty_living_in_families_below_the_poverty_line','uninsured','production_transportation_and_material_moving_occupations',
                      'poverty_rate_below_federal_poverty_threshold','construction_extraction_maintenance_and_repair_occupations',
                      'less_than_high_school','at_least_bachelor_s_degree','adults_65_and_older_living_in_poverty','unemployment','graduate_degree',
                      'latino_population','management_professional_and_related_occupations','sire_homogeneity','native_american_population',
                      'sales_and_office_occupations','white_not_latino_population']].corr()
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
    f, ax = plt.subplots(figsize=(15, 11))

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
big_df['sire_norm'] = (big_df.sire_homogeneity - big_df.sire_homogeneity.mean()) / big_df.sire_homogeneity.std()
sns.jointplot(x=big_df.sire_norm,y=big_df['white_not_latino_population'],data=all_df,kind='reg')


big_df.sort_values(by='avg_percent',ascending=False)
big_df.avg_percent.plot.hist()
big_df.low_turnout.plot.hist()
big_df.gini_coefficient.plot.hist()
big_df.size_rank.plot.hist()

###Calculate PCA and plot results
## Needs more work
from sklearn.decomposition import PCA
pca = PCA(n_components=16)
pca_data = pca.fit_transform(xTrain)
pca.explained_variance_ratio_
eig_values = pca.explained_variance_
eig_vectors = pca.components_
pc1 = pca.components_[0]
pc2 = pca.components_[1]
pc3 = pca.components_[2]
pc4 = pca.components_[4]
pc5 = pca.components_[5]


index= big_df.columns
structure_loading_1 = pc1* np.sqrt(eig_values[0])
str_loading_1 = pd.Series(structure_loading_1, index=index)
str_loading_1

sns.barplot(index,pca.explained_variance_ratio_)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
