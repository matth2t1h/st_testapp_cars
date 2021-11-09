### st_testapp_cars.py
### Streamlit app for displaying data about cars

### starting the app: inside the directory containing the Python script, run the following command:
### streamlit run <my_script.py>
### streamlit run st_testapp_cars.py

###================================
### PACKAGES

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import streamlit as st

###================================
### FUNCTIONS

def createHeatmap(corrDF, x, y, tmpTitle, tmpAx):
  ### correlation heatmap creation
  
  mask = np.triu(np.ones_like(corrDF, dtype=bool))                              ### displaying the bottom left corner
  
  tmp_htmp = sns.heatmap(
      data=corrDF,
      cmap="vlag",
      # ax=axs[x,y],
      square=True,
      vmin=-1,
      vmax=1,
      annot=True,
      mask=mask,
      ax=tmpAx
  )

  tmp_htmp.set_title(tmpTitle)

  return tmp_htmp

def dfForMasking(origDF,thres=0):
  ### correlation dataframe creation, with a mask based on the given threshold
  endDF = origDF.corr()
  endDF = endDF[ np.abs(endDF) >= thres ]

  return endDF

def get_means_Continent(origDF, colList, tmpContinent):

  tmpDF = origDF[colList].loc[ origDF['continent'] == tmpContinent ].groupby(['year']).mean().reset_index()

  for tmpCol in tmpDF.columns:
    if tmpCol != 'year':
      tmpDF[tmpCol] = tmpDF[tmpCol].apply(lambda x: round(x, 2))

  return tmpDF

###================================
### MAIN PROGRAM

###-----------------------
### dataset import

link = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv"
dfcars = pd.read_csv(link)
dfcars.head()

###-----------------------
### preprocessing

dfcars['continent'] = dfcars['continent'].apply(lambda l: l.replace('.','').replace(' ',''))
dfcars.sort_values(by=['year'], inplace=True)
dfcars['mpg'] = dfcars['mpg'].apply(lambda l: round(l,2))

###-----------------------
### first analysis

dfMeansYearsCars = dfcars[['year','mpg','hp','cylinders','cubicinches','weightlbs']].groupby(['year']).mean().reset_index()

for tmpCol in dfMeansYearsCars.columns:
  if tmpCol != 'year':
    dfMeansYearsCars[tmpCol] = dfMeansYearsCars[tmpCol].apply(lambda x: round(x, 2))

### counting the number of cars per year
dfCountYears = dfcars['year'].value_counts().reset_index()
dfCountYears.columns = ['year','counts']
dfCountYears.sort_values(by='year', ascending=True, inplace=True)
dfCountYears.reset_index(inplace=True, drop=True)

fig_countYears = px.bar(
    dfCountYears,
    x='year',
    y='counts'
)

fig_countYears.update_xaxes(type='category')

# fig_countYears.show()

### computing the mean of each variable, grouping by year
listColumns = ['year','continent','mpg','hp','cylinders','cubicinches','weightlbs']
listColumnsNoYearCont = ['mpg','hp','cylinders','cubicinches','weightlbs']
dfMeansYearsCars_US = get_means_Continent(dfcars, listColumns, 'US')
dfMeansYearsCars_Eur = get_means_Continent(dfcars, listColumns, 'Europe')
dfMeansYearsCars_Jap = get_means_Continent(dfcars, listColumns, 'Japan')

###================================
### STREAMLIT INTERFACE

st.title('Analysis')
st.write('on the dataset about cars')

with st.container():
    
    col_titleDataset, col_empty, col_expandDataset = st.columns([3,1,8])

    with col_titleDataset:
        st.subheader("Raw data:")
    with col_expandDataset:
        with st.expander(label='', expanded=False):
            st.dataframe(data=dfcars)                               #### , width=None, height=None

with st.container():
    
    col_titleDataset, col_empty, col_expandDataset = st.columns([3,1,8])

    with col_titleDataset:
        st.subheader("Correlation heatmap:")
    with col_expandDataset:
        with st.expander(label='', expanded=False):
            
            allCorr = dfForMasking(dfMeansYearsCars)

            fig, axs = plt.subplots(figsize=(8,8)) 
            
            mask = np.triu(np.ones_like(allCorr, dtype=bool))                              ### displaying the bottom left corner
    
            htmp = sns.heatmap(
                data=allCorr,
                cmap="vlag",
                square=True,
                vmin=-1,
                vmax=1,
                annot=True,
                mask=mask,
                ax=axs
            )

            htmp.set_title("Correlation heatmap - Variables about cars")
            
            st.write(fig)

with st.container():

    col_rButtonsHtmp, col_empty, col_explain = st.columns([3,1,8])

    with col_rButtonsHtmp:
        sel_explainType = st.radio(
            "Variable to explain:",
            options=listColumnsNoYearCont
        )

    with col_explain:
        
        if sel_explainType == 'mpg':
            st.markdown('Miles per gallon: strongely positively correlated with the years. It means that the car consumption has become more efficient, it is lower nowadays than in 1971.')
            st.markdown(
                """
                But this variable is strongly negatively correlated with the others. It means that the more distance we can travel per gallon, the less :
                - cylinders we have,
                - horse power we have,
                - weight we have,
                - cubic inches we have.
                """
            )
        elif sel_explainType == 'hp':
            st.markdown('Horse power: strongely positively correlated with the car weight, the cubic inches and the cylinders. The more weight/cubic inches/cylinders we have, the more powerful the car will be.')
            st.markdown('But this variable is strongely negatively correlated with the years, so it means that the cars were more powerful before than nowadays.')
        elif ((sel_explainType == 'cylinders') or (sel_explainType == 'cubicinches') or (sel_explainType == 'weightlbs')):
            st.markdown('\'' + sel_explainType + '\': same analysis as for the horse power (\'hp\').')
        else:
            st.markdown('')

with st.container():
    
    col_titleDataset, col_empty, col_expandDataset = st.columns([3,1,8])

    with col_titleDataset:
        st.subheader("Analysis per continent:")
    with col_expandDataset:
        sel_continent = st.selectbox(
            "Filter the analysis on one continent:",
            options=dfcars['continent'].unique()
        )

        dfContinent = pd.DataFrame()

        if sel_continent == 'US':
            dfContinent = dfMeansYearsCars_US.copy()
        elif sel_continent == 'Europe':
            dfContinent = dfMeansYearsCars_Eur.copy()
        elif sel_continent == 'Japan':
            dfContinent = dfMeansYearsCars_Jap.copy()
        else:
            dfContinent = pd.DataFrame()

### radio buttons: Distribution, mpg, hp, cylinders, cubicinches, weightlbs
with st.container():

    col_rButtons, col_graph = st.columns(2)

    with col_rButtons:
        sel_chartType = st.radio(
            "Variable to display:",
            options=listColumnsNoYearCont
        )

    with col_graph:

        fig = px.line(
            dfContinent,
            x='year',
            y=sel_chartType
        )
        fig.update_xaxes(type='category')
        st.write(fig)
