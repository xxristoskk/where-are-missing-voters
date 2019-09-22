## Reshape the main data frame and add new/remove features
df_eda = pd.DataFrame(df.drop(['voter_percent','2018_vote_percent'],axis=1))
df_eda['avg_percent'] = (df['voter_percent'] + df['2018_vote_percent']) / 2
df_eda['low_turnout'] = [1 if x <= .38 else 0 for x in df_eda['avg_percent']]

target = df_eda['low_turnout'].copy()
features = pd.DataFrame(df_eda[['gini_coefficient','size_rank','asian_american_population','african_american_population',
                      'uninsured','production_transportation_and_material_moving_occupations',
                      'poverty_rate_below_federal_poverty_threshold','construction_extraction_maintenance_and_repair_occupations',
                      'less_than_high_school','at_least_bachelor_s_degree','unemployment',
                      'latino_population','management_professional_and_related_occupations','sire_homogeneity','native_american_population',
                      'sales_and_office_occupations','white_not_latino_population']])


### EDA plotting and prinicpal component analysis
#Pairplot of most influential features for voter turnout
pair_feats = ['gini_coefficient','less_than_high_school','at_least_bachelor_s_degree','unemployment','latino_population',
            'white_not_latino_population','asian_american_population','african_american_population',]
n = 4
row_groups= [pair_feats[i:i+n] for i in range(0, 4, n)]
for i in row_groups:
    pp = sns.pairplot(data=df_eda,y_vars='avg_percent',x_vars=i ,kind="reg", height=4)

##Correlated features
corr = features.corr()

df_eda.sort_values(by='avg_percent',ascending=False)
df_eda.avg_percent.plot.hist()
df_eda.low_turnout.plot.hist()
df_eda.gini_coefficient.plot.hist()
df_eda.size_rank.plot.hist()

###Calculate PCA and plot results
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
pca_data = pca.fit_transform(X)
df_pca = pd.DataFrame(data=pca_data,columns=['PC1','PC2','PC3','PC4','PC5'])
result_df = pd.concat([df_pca, target],axis=1)

plt.scatter(pca_data[:,0],pca_data[:,1])
sns.barplot(pca.explained_variance_ratio_,index)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
