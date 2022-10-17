import streamlit as st
#from streamlit import caching
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import math
from scipy import stats
from scipy.stats import shapiro
from PIL import Image

import seaborn as sns
from sklearn.cluster import KMeans
import plotly.graph_objects as go


import base64
# Set page configuration
st.set_page_config(page_title ="Statistical Customer Segmentation",
                    initial_sidebar_state="expanded",
                    layout='wide',
                    page_icon="🛒")


# Page Title 
#introduction()
logo = Image.open('c5i_logo.jpg')

st.sidebar.image(logo,use_column_width=True)

# Set up the page
@st.cache(persist=False,
          allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner= True)


# Preparation of data
def prep_data(df):
    col = df.columns
    return col


#########################
# Load data from user upload
def load_upload(input):
    # Create Data Frame
    df = pd.read_csv(input,sep=None ,engine='python', encoding='utf-8',
                parse_dates=True,
                infer_datetime_format=True, index_col=0)

    return df

# Load data from the local repository 
def load_local():
    # Create Data Frame
    df_abc = pd.read_csv('data/Mall_Customers.csv',sep=None ,engine='python', encoding='utf-8',
                parse_dates=True,
                infer_datetime_format=True)
    df = pd.read_csv('data/Mall_Customers.csv',sep=None ,engine='python', encoding='utf-8',
                parse_dates=True,
                infer_datetime_format=True, index_col=0)
    
    return df_abc, df

def upload_ui():
    st.sidebar.subheader('Load a Dataset 💾')
    st.sidebar.write("Upload your dataset (.csv)")
    # Upload
    input = st.sidebar.file_uploader('')
    if input is None:
        dataset_type = 'LOCAL'
        #st.sidebar.write("_If you do not upload a dataset, an example is automatically loaded to show you the features of this app._")
        df_abc, df = load_local()
        list_var= dataset_ui(df_abc, df, dataset_type)
    else:
        dataset_type = 'UPLOADED'
        with st.spinner('Loading data..'):
            df = load_upload(input)
            #st.write(df.head())
            df_abc = pd.DataFrame()
        list_var= dataset_ui(df_abc, df, dataset_type)
 


    # Process filtering
    st.write("\n")
    st.subheader('''📊 Your dataset with the final version of the features''')
    df = df[list_var].copy()
    st.write(df.head(2))

    return list_var, dataset_type, df, df_abc

def dataset_ui(df_abc, df, dataset_type):
    # SHOW PARAMETERS
    expander_default = (dataset_type=='UPLOADED')
    
    st.subheader('🛎️ Please choose the following features in your dataset')
    with st.expander("FEATURES TO USE FOR THE ANALYSIS"):
        st.markdown('''
        _Select the columns that you want to include in the analysis of your sales records._
    ''')
        dict_var = {}
        for column in df.columns:
            dict_var[column] = st.checkbox("{} (IN/OUT)".format(column), value = 1)
    filtered = filter(lambda col: dict_var[col]==1, df.columns)
    list_var =list(filtered)



    return list_var

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
    
def export_ui(df_abc):
    st.header('**Export results **')
    #st.header('**Export results ✨**')
    st.write("_Finally you can export the results of your segmentation with all the parameters calculated._")
    if st.checkbox('Export Data',key='show2'):
        with st.spinner("Exporting.."):
            st.write(df_abc.head())
            #df_abc = df_abc.to_csv(decimal=',').encode()
            #b64 = base64.b64encode(df_abc).decode()
            #href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a>'
            #st.download_button("Press to Download",df_abc, "file.csv", "text/csv", key='download-csv') 
            #st.markdown(href, unsafe_allow_html=True)
            df_abc1 = convert_df(df_abc)

            st.download_button(
                label="Download data as CSV",
                data=df_abc1,
                file_name='Preprocessed_df.csv',
                mime='text/csv',
            )

# Upload Data Set
list_var,dataset_type, df, df_abc = upload_ui()



# Start Calculation ?
#if st.checkbox('Start Calculation',key='show', value=False):
 #   start_calculation = True
#else:
 #   if dataset_type == 'LOCAL':
  #      start_calculation = True
   # else:
    #    start_calculation = False

# Process df_abc for uploaded dataset
#if dataset_type == 'UPLOADED' and start_calculation:
if dataset_type == 'UPLOADED' or dataset_type =='LOCAL':

    st.header("**Elbow point 💹**")
    #df_abc, wcss  = abc_processing(df)
    #print(wcss)
    col1,col2 = st.columns(2)
    
    #Choosing the Annual Income Column & Spending Score column
    X = df.iloc[:,[2,3]].values
    # finding wcss value for different number of clusters
    wcss = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    
    

    x_axis = list(range(0, 11))
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=x_axis,y=wcss,mode='lines',line=dict(color='blue')))
    fig.update_layout(xaxis_title='<b>No. of clusters</b>',yaxis_title='<b>WCSS</b>',title='<b>The Elbow Point Graph</b>',title_x=0.5)
    st.plotly_chart(fig,use_container_width=True)

    st.write('optimum number of cluster = 4')
    
    st.header("**Clusters**")
    #Training the k-Means Clustering Model
    kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0)

    # return a label for each data point based on their cluster
    Y = kmeans.fit_predict(X)

    #Visualizing all the Clusters
    # plotting all the clusters and their Centroids

    fig1=go.Figure()
    fig1.add_trace(go.Scatter(x=X[Y==0,0],y=X[Y==0,1],mode='markers',line=dict(color='green'), name='Cluster 1'))
    fig1.add_trace(go.Scatter(x=X[Y==1,0],y=X[Y==1,1],mode='markers',line=dict(color='red'), name='Cluster 2'))
    fig1.add_trace(go.Scatter(x=X[Y==2,0],y=X[Y==2,1],mode='markers',line=dict(color='yellow'), name='Cluster 3'))
    fig1.add_trace(go.Scatter(x=X[Y==3,0],y=X[Y==3,1],mode='markers',line=dict(color='violet'), name='Cluster 4'))
#    fig1.add_trace(go.Scatter(x=X[Y==4,0],y=X[Y==4,1],mode='markers',line=dict(color='blue'), name='Cluster 5'))
    fig1.update_layout(xaxis_title='<b>Annual Income</b>',yaxis_title='<b>Spending Score</b>',title='<b>Customer Groups</b>',title_x=0.5)
 
    # plot the centroids
    fig1.add_trace(go.Scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],mode='markers',line=dict(color='cyan'), name='Centroids'))
    st.plotly_chart(fig1,use_container_width=True)
   
else:
    list_sku = ['Gender',	'Age',	'Annual Income (k$)',	'Spending Score (1-100)']

# Start Calculation after parameters fixing
#if start_calculation:

    # Part 1: Elbow point graph
#    st.header("**Elbow point 💹**")

    # Part 2: Clusters

    # Part 5: Export Results
    #export_ui(df_abc)



