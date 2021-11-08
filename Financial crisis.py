from pandas_datareader import data
import plotly.express as px
from sklearn.metrics import classification_report
# Plotting 
import matplotlib.pyplot as plt
import seaborn
import matplotlib.mlab as mlab

# Statistical calculation
from scipy.stats import norm


from numpy import log as ln
import numpy as np
from math import *
import datetime
import seaborn as sn
import seaborn as sns # data visualization library  
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std


from scipy.stats import shapiro



import pandas as pd 


from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import plotly.figure_factory as ff
import scipy.cluster.hierarchy as shc
import numpy as np
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib



from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import euclidean_distances


SP_500_data = pd.read_csv('SP_500_data_commodities_data.csv')
SP_500_data=SP_500_data.set_index('Date')
SP_500_data=SP_500_data.drop(columns="Unnamed: 0")

SP_500_data=SP_500_data.loc["2005-01-01":]

SP_500_data_return=SP_500_data.pct_change()[1:-1]
#SP_500_data_return=SP_500_data_return.fillna(0)
SP_500_data_return.head()

def correlations_plot_and_data(year, month_start, month_end):

  data_m =data_to_cluste_month(SP_500_data_return,year,month_start,month_end)
  data_m=data_m.dropna(axis=1)
  correlations = data_m.corr()
  #correlations[np.arange(correlations.shape[0])[:,None] > np.arange(correlations.shape[1])] = np.nan
  #ndf = correlations.unstack().to_frame().T
  #ndf.columns = ndf.columns.map('{0[0]}_{0[1]}'.format) 
  #ndf.index=[data_m.index[0]]




  fig = px.imshow(correlations,
                labels=dict(x="Russell_Commodities", y="Russell_Commodities", color="correlations"),
                x=correlations.index.to_list(),
                y=correlations.columns.to_list()
               )
  fig.update_layout(
      autosize=False,
      width=1000,
      height=1000,)
  fig.show() 

  return correlations


def correlations_plot_and_data__2(year, month_start, month_end):

  data_m =data_to_cluste_month(SP_500_data_return,year,month_start,month_end)
  #data_m=data_m.dropna(axis=1)
  #data_m=data_m.fillna(0)  
  correlations = data_m.corr(method='pearson')
  #correlations[np.arange(correlations.shape[0])[:,None] > np.arange(correlations.shape[1])] = np.nan
  #ndf = correlations.unstack().to_frame().T
  #ndf.columns = ndf.columns.map('{0[0]}_{0[1]}'.format) 
  #ndf.index=[data_m.index[0]]


  return correlations


def data_month_just_one(rendement_all_years,year,month):
      response_year1 =year
      response_year2 = year
      array=[]
      for i in range(int(response_year1),int(response_year2)+1):
        array.append(i)
      rendement_all_years['year'] = pd.DatetimeIndex(rendement_all_years.index).year
      
      dataf=rendement_all_years[rendement_all_years['year'].isin(array)]  

      response_month1 = month
      response_month2 = month
      array=[]
      for i in range(int(response_month1),int(response_month2)+1):
        array.append(i)
      dataf['month'] = pd.DatetimeIndex(dataf.index).month
      
      datafv=dataf[dataf['month'].isin(array)]  
      
      datafv=datafv.drop(columns=["year"])
      datafv=datafv.drop(columns=["month"])
   
    
      return datafv

def data_to_cluste_month(rendement_all_years,year,month_start, month_end):
      response_year1 =year
      response_year2 = year
      array=[]
      for i in range(int(response_year1),int(response_year2)+1):
        array.append(i)
      rendement_all_years['year'] = pd.DatetimeIndex(rendement_all_years.index).year
      
      dataf=rendement_all_years[rendement_all_years['year'].isin(array)]  

      response_month1 = month_start
      response_month2 = month_end
      array=[]
      for i in range(int(response_month1),int(response_month2)+1):
        array.append(i)
      dataf['month'] = pd.DatetimeIndex(dataf.index).month
      
      datafv=dataf[dataf['month'].isin(array)] 
      
      datafv=datafv.drop(columns=["year"])
      datafv=datafv.drop(columns=["month"])
    
    
      return datafv
     
     
print("ok")


# just one fig
dat_test=data_to_cluste_month(SP_500_data_return,2015,2,7)
#dat_test=dat_test.fillna(0)

#plt.figure(figsize=(15,10))
correlations = dat_test.corr(method='pearson')
#sns.heatmap(round(correlations,2),   cmap="PiYG",)

# cmap="Blues"
# cmap="YlGnBu"
# cmap="BuPu"
# cmap="Greens"
# cmap="PiYG"


fig = px.imshow(correlations,
            labels=dict(x="SP_500_data_commodities_data", y="SP_500_data_commodities_data", color="correlations"),
            x=correlations.index.to_list(),
            y=correlations.columns.to_list()
            )
fig.update_layout(
    autosize=False,
    width=700,
    height=800,)
fig.show() 
#fig.savefig("SP_500_data_commodities_data.png",dpi=120) 
fig.write_image(file='"SP_500_data_commodities_data.png",', format='.png')

