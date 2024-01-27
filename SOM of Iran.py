# -*- coding: utf-8 -*-
"""
This is a SOM for iran provinces
Ali Asghari
"""
import pandas as pd
from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import glob
import os

#getting all datas as seperate DFs and then merge them into one dataframe

all_files1 = glob.glob("./data/" + "/*.csv")
all_files1.sort(key=os.path.getmtime)
List_DATA = []

for filename in all_files1:
    data = pd.read_csv(filename, index_col=None)  
    List_DATA.append(data)
    

glued_data = List_DATA[0]
for i in range(1,len(List_DATA)):
    glued_data = pd.merge(glued_data, List_DATA[i], on="Province")
    
    
    

data_x  = glued_data.iloc[:, 1:].values
data_y = glued_data.iloc[:, 0].values


#Creating the grid and train the data

feature_names = ['Unemployment Rate', '1st_Class_Pass_Ratio', 'Left_Education_Ratio', 'Cinema_per_person',
 'Library_per_person', 'Hospitalbed_per_person', 'Rural_Health_Centers_per_person', 'Life Expectancy', 'Healthy_Bathroom_Rural',
 'Healthy_Bathroom_Urban', 'Healthy_Kitchen_Rural', 'Healthy_Kitchen_Urban', 'Gini_index_Rural', 'Gini_index_Urban',
 'Pipeline_Water_Rural', 'Electricity_Rural', 'Pipeline_Gas_Rural', 'Pipeline_Water_Urban',
 'Pipeline_Gas_Urban', 'Urban_Housing/Total_Expenitures', 'Rural_Housing/Total_Expenditures', 'Inflation_Urban', 'Inflation_Rural', 'Net_Revenue_Uran', 'Net_Revenue_Rural', 'in_city_accidents',
 'Drug_arrested', 'Accident_Death_ratio', 'Robbery_Arrested', 'Crime_Arrested', 'Divorece', 'Election_Participation',
 'Rain_Deviation', 'Absolute_Difference_From_Equality_in_Students_Gender']



from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
data_x = sc.fit_transform(data_x)
size = 10
som = MiniSom(10, 10, data_x.shape[1],neighborhood_function='gaussian', sigma=1.5,
random_seed=1)
som.pca_weights_init(data_x)
som.train_random(data_x, 10000, verbose=True)






#create Dendogram to estimate the number of cluster

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(data_x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()







# Training the Hierarchical Clustering model on the dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(data_x)
result = np.column_stack((y_hc, data_y))


# Add color and labels 
category_color = {
                  0: 'green',
                  1: 'red',
                  2: 'blue',
                  3: 'crimson',
                  4: 'black',
                  5: 'orange'
                  }

Provs=['E_Azerbaijan','W_Azerbaijan','Ardabil','Isfahan','Alborz','Ilam','Bushehr','Tehran','Chaharmahal&Bakhtiari','S_Khorasan',
       'R_Khorasan','N_Khorasan','Khuzestan','Zanjan','Semnan','Sistan&Baluchestan',
       'Fars','Qazvin','Qom','Kurdistan','Kerman','Kermanshah','Kohgiluyeh&Boyer-Ahmad','Golestan','Gilan','Lorestan',
       'Mazandaran','Markazi','Hormozgan','Hamadan','Yazd']
Iran_map = som.labels_map(data_x,Provs)


colors_dict = {c: category_color[dm] for c, dm in zip(Provs,y_hc)}



#plot The SOM
    
plt.figure(figsize=(30, 30))
for p, Provinces in Iran_map.items():
    Provinces = list(Provinces)
    x = p[0] + .001
    y = p[1] - .003
    for i, c in enumerate(Provinces):
        off_set = (i+1)/len(Provinces) - 0.5
        plt.text(x, y+off_set, Provinces[i],color=colors_dict[c], fontsize=25)
plt.pcolor(som.distance_map().T, cmap='gray_r', alpha=1)
plt.xticks(np.arange(size+1))
plt.yticks(np.arange(size+1))
plt.grid()



# Plot the heatmap for all indicators

W = som.get_weights()
plt.figure(figsize=(30, 30))
for i, f in enumerate(feature_names):
    plt.subplot(6, 6, i+1)
    plt.title(f)
    plt.pcolor(W[:,:,i].T, cmap='coolwarm')
    plt.xticks(np.arange(size+1))
    plt.yticks(np.arange(size+1))
plt.tight_layout()
plt.show()


# Calculate the importance of GDP share usigng DecisionTreeRegressor

GDP = pd.read_csv('./data/GDP_share/GDP_share.csv', index_col=None)  
glued_data2 = pd.merge(glued_data, GDP, on="Province")



feature_names2 = ['Unemployment Rate', '1st_Class_Pass_Ratio', 'Left_Education_Ratio', 'Cinema_per_person',
 'Library_per_person', 'Hospitalbed_per_person', 'Rural_Health_Centers_per_person', 'Life Expectancy', 'Healthy_Bathroom_Rural',
 'Healthy_Bathroom_Urban', 'Healthy_Kitchen_Rural', 'Healthy_Kitchen_Urban', 'Gini_index_Rural', 'Gini_index_Urban',
 'Pipeline_Water_Rural', 'Electricity_Rural', 'Pipeline_Gas_Rural', 'Pipeline_Water_Urban',
 'Pipeline_Gas_Urban', 'Urban_Housing/Total_Expenitures', 'Rural_Housing/Total_Expenditures', 'Inflation_Urban', 'Inflation_Rural', 'Net_Revenue_Uran', 'Net_Revenue_Rural', 'in_city_accidents',
 'Drug_arrested', 'Accident_Death_ratio', 'Robbery_Arrested', 'Crime_Arrested', 'Divorece', 'Election_Participation',
 'Rain_Deviation', 'Absolute_Difference_From_Equality_in_Students_Gender','GDP_per_capit_share']



X = glued_data2.iloc[:, 1:].values
X = sc.fit_transform(X)


feature_df = pd.DataFrame(X, columns=feature_names)
target = feature_df.iloc[:,-1]
Features = feature_df.iloc[:,0:-1]

# fit the data to the regressor
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=0)
model.fit(Features, target)

# Get importance and feature names
importances = model.feature_importances_
feature_names = Features.columns

# Create dataframe with features and importances
feat_importances = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort by importance and slice top 4
feat_importances = feat_importances.sort_values('importance', ascending=False).iloc[:6] 

# Plot top 4
feat_importances.plot(x='feature', y='importance', kind='barh', title='Feature Importances')
plt.show()








