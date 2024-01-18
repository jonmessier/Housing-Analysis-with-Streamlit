import streamlit as st
import pandas as pd
import plotly.graph_objects as go 
import plotly.express as px
#Mapping tools
import folium
from streamlit_folium import st_folium

import datetime as datetime

APP_TITLE = "Home Features and Sales Price"
APP_SUB_TITLE = "Choose a feature to see Neural Network clustering data"

#Clustering colormap
colormap ={0:'blue',1:'red',2:'green', 3:'yellow', 4:'purple'}



def feature_select(df):
     #Show a sidebar to 
     st.sidebar.write("Choose the feature data to compare the unsupervised clustering comparison.")
     feature = st.sidebar.selectbox('Feature selection',df.columns[1:])
     fig = px.box(df, y=feature, x='cluster')
     st.plotly_chart(fig)
     return feature

def select_clusters(df):
     st.sidebar.write("Choose the clusters to compare their locations on the map.")
     cluster_list = list(df['cluster'].unique())
     cluster_list.sort()
     clusters = st.sidebar.multiselect('Select Clusters to map',cluster_list)
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
     st.set_page_config(APP_TITLE)
     st.title(APP_TITLE)
     st.caption(APP_SUB_TITLE)

     #Import Data
     df = pd.read_csv("House_listing_data.csv")
     #Rename columns to work with folium
     df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"}, inplace=True)

     feature = feature_select(df)

     clusters = select_clusters(df)
     display_map(df,clusters)

if __name__ == "__main__":
    main()
