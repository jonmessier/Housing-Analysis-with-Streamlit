import streamlit as st
import pandas as pd

#import plotly.graph_objects as go 
import plotly.express as px

#Mapping tools
import folium
from streamlit_folium import st_folium

import datetime as datetime

st.title('Housing Data Analysis')
tab_titles = ['Overview', "Clustering", "Price Modeling"]
tab1, tab2, tab3 = st.tabs(tab_titles)

#Clustering colormap
colormap ={0:'blue',1:'red',2:'green', 3:'yellow', 4:'purple'}



def feature_select(df):
     st.write("Choose a feature to see a comparison between clusters.")
     feature = st.selectbox('Feature selection',df.columns[1:])
     fig = px.box(df, y=feature, x='cluster')
     st.plotly_chart(fig)
     return feature

def select_clusters(df):
     cluster_list = list(df['cluster'].unique())
     cluster_list.sort()
     clusters = st.multiselect('Select Clusters to map',cluster_list)
     return clusters

def display_map(df, clusters):
     #location of 1st data point in dataframe
     lat1 = df['latitude'].iloc[0]
     long1 = df['longitude'].iloc[0]

     # center on 1st item in the dataframe
     m = folium.Map(location=[lat1,long1], zoom_start=10)
     for cluster in clusters:
          cluster_data = df.loc[df['cluster']==cluster]
          for i in range(0,len(cluster_data)):
               folium.Marker(
                    location = [cluster_data.iloc[i]['latitude'], cluster_data.iloc[i]['longitude']],
                    icon=folium.Icon(color=colormap[cluster])
                    ).add_to(m)

     # call to render Folium map in Streamlit
     st_data = st_folium(m, width=725)


def main():
     #Import Data
     df = pd.read_csv("House_listing_data.csv")
     #Rename columns to work with folium
     df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}, inplace=True)

     with tab1:
          st.markdown('''
               # Problem Statement
               A house flipping company would like to identify underpriced homes by comparing asking prices to predicted sale prices.   They would like to segment homes into groups to analyze what kinds of homes there are. They would also like a model that predicts the selling price of a home.

               The company has provided publicly available data from the King County Assessor's office to use for clustering and prediction.
               
               **Data notes:**  
               Data can be found [HERE](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2Fe%2F2PACX-1vSz2GYT82APb1iS8n07ZXi5C3WsKNEZ7lZu2RPUfHi_ZDZb1A2tmBIQyuQvJf9GbHgTNp2WXj2H6ZHC%2Fpub%3Foutput%3Dcsv)  
               BrickStone is the percentage of a house that is made of brick or stone.

               # KMeans Clustering  
               To help our customer out, we are going to group the houses into clusters using the KMeans algorithm.  The optimal number of clusters will be determined by observing the Elbow plot and Silhouette scores.  
               
               K-means clustering is a popular unsupervised machine learning algorithm used for partitioning a dataset into distinct, non-overlapping subsets or clusters. The goal of K-means is to group similar data points into the same cluster while keeping different clusters as dissimilar as possible.  

               K-means is sensitive to the initial placement of centroids, and different initializations may lead to different results. To mitigate this, multiple runs with different initializations are often performed, and the solution with the lowest total intra-cluster variance is chosen.

               ## Optomizing KMeans with Elbow Plot and Silhouette scores
               When evaluating an Elbow plot, I am looking for dramatic (elbow-like) bends in the inertia/clusters ratio. Here, the bends are not very clear. If I had to guess I would say at 4 clusters we see a bend. Let's verify with a Silhoutte score plot.
          ''',unsafe_allow_html=True)     
               
          st.image("elbowplot.png", caption="Elbow Plot")

          st.markdown('''
               When analyzing Silhouette scores we are looking for the number of clusters that produces the highest score. Where our elbow plot was ambiguous the Sil score plot has a distinct peak at n_clusters=5
          ''',unsafe_allow_html=True)

          st.image("silhouettescore.png", caption="Silhouette Score")

     with tab2:
          st.header("Unsupervised Clustering")
          st.write("KMeans Clustering algorithm is an iterative algorithm that tries to partition the dataset into K pre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. It tries to make the intra-cluster data points as similar as possible while also keeping the clusters as different (far) as possible. It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more homogeneous the data points are within the same cluster.")
          st.divider()
          feature = feature_select(df)
          st.divider()
          st.write("Let's take a look at where these clusters of houses appear on a map.  Choose a cluster or clusters to show.  Can you make some observations to describe each cluster and their relative locations?")
          clusters = select_clusters(df)
          display_map(df,clusters)

          st.write('''
          ## Observations
          This is a lot of intersting data!  Comparing all the features for every cluster would take a lot of time.  Instead, I will try to highlight details which make these clusters unique.  Of course this would be modified through communication with the stakeholder identifying which features they deem important.

          ### Cluster 0
          - 2nd highest median sale price.  Median = $736k
          - 3rd highest total living sq ft.
          - Almost exclusively represented by Brickstone fascade
          - Built between 1940 and 1955

          ### Cluster 1
          - 3rd highest median sale price. Median = $688K
          - Overall low sq.ft. with lowest sq.ft on 1st, and 2nd floor, but having the most sq ft. on the 3rd floor compared to other clusters.  Are these row-homes?
          - Lowest number of bedrooms with a median of 2.  Per
          - Newest construction of any cluster with a median year built of 2017

          ### Cluster 2
          - 4th highest sale price. Median = $650k
          - 2nd highest total sq ft. Mdeian = 2210sqft
          - Largest overall finished basement. Median = 780sq.ft.
          - 2nd highest median bedrooms:4

          ### Cluster 3
          - Lowest Median price $485K
          - Lowest total sq ft. Median = 1410
          - 3 bedroom / 1 bath house.  low variation and few outliers to this description
          - built between 1900 and early 2000s

          ### Cluster 4
          - Highest median price $775K
          - Largest total living space and largest 2nd floor space.
          - Largest attached garages. Median 470sqft
          - 2nd newest houses built from 1990-present
          - Highest bedroom count. Median=4
          - highest median bathroom count. Median = 2
          ''')

     with tab3:
          st.header("Price modeling")
          st.write("The house flipping company has provided new data about when the homes were sold and what businesses are nearby. This new data has created a very large number of columns as the businesses, months sold, and years sold are all one-hot encoded.")
          st.divider()
          st.write('''
          Using a Neural Network described by:
          - 1 input layer with 10 Densely connected nodes, RELU activation
          - 1 hidden layer with 10 densely connected nodes, RELU activation, and L1 and L2 regularization
          - Dropout 20%
          - 1 hidden layer with 10 densely connected nodes, RELU activation
          - 1 Output layer with linear activation

          The following results were achieved:

          |Model | MSE | RMSE | MAE | R^2 |
          | :---| :---|:---|:---|:---|
          |Model_4 Train|       2.969582e+10|  172324.761641 | 123109.764054|  0.667170|
          |Model_4 test|        4.455284e+10 | 211075.431029|  150082.983641 | 0.490218|
          ''')

          st.image("NN_loss.png")
          st.image("NN_MAE.png")
          st.image("NN_RSME.png")

          st.write('''
          This model provides the highest $R^2$ value and combined with the lowest combined RMSE score. The $R^2$ value is telling us is that the model accounts for \~49% of the variance in the data.  In general, a model with a higher R^2 value closer to 1 is better, however an $R^2=1$ usually indicates overfitting.  A SME is needed to verify the strength of this metric for the dataset.  

          The RMSE is telling us the root mean square absolute error (more simply thought of as the average error from the mean without +\/-) is about $211k.  Since our mean home price is $664,496.90, this error represents nearly \~+\\-30% of the value. This may or may not be helpful to our client depending on their level of financial risk and understanding of the property value.  If you could purchase a property below the predicted value less the RSME you have a high likihood of a potential profit.  Of course, the house would have to be resold and factors like interest paid, taxes, repairs are not accounted for!  This is Not Financial Advice!
          
          ''')

if __name__ == "__main__":
    main()
