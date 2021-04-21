#!/usr/bin/env python
# coding: utf-8

# <a id='Q0'></a>
# <center><a target="_blank" href="http://www.propulsion.academy"><img src="https://drive.google.com/uc?id=1McNxpNrSwfqu1w-QtlOmPSmfULvkkMQV" width="200" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <center> <h4 style="color:#303030"> Python for Data Science, Homework, template: </h4> </center>
# <center> <h1 style="color:#303030">Breast Cancer Selection</h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center style="color:#303030"><h4>Propulsion Academy, 2021</h4></center>
# <p style="margin-bottom:1cm;"></p>
# 
# <div style="background:#EEEDF5;border-top:0.1cm solid #EF475B;border-bottom:0.1cm solid #EF475B;">
#     <div style="margin-left: 0.5cm;margin-top: 0.5cm;margin-bottom: 0.5cm">
#         <p><strong>Goal:</strong> Binary classification on Breast Cancer data</p>
#         <strong> Sections:</strong>
#         <a id="P0" name="P0"></a>
#         <ol>
#             <li> <a style="color:#303030" href="#SU">Set Up </a> </li>
#             <li> <a style="color:#303030" href="#P1">Modeling</a></li>
#             <li> <a style="color:#303030" href="#P2">Evaluation</a></li>
#         </ol>
#         <strong>Topics Trained:</strong> Binary Classification.
#     </div>
# </div>
# 
# 
# <nav style="text-align:right"><strong>
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/" title="momentum"> SIT Introduction to Data Science</a>|
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/weeks/week2/day1/index.html" title="momentum">Week 2 Day 1, Applied Machine Learning</a>|
#         <a style="color:#00BAE5" href="https://colab.research.google.com/drive/1DK68oHRR2-5IiZ2SG7OTS2cCFSe-RpeE?usp=sharing" title="momentum"> Assignment, Classification of the success of pirate attacks</a>
# </strong></nav>

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# In[ ]:


# !sudo apt-get install build-essential swig
# !curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
# !pip install auto-sklearn
# !pip install pipelineprofiler # visualize the pipelines created by auto-sklearn
# !pip install shap
# !pip install --upgrade plotly
# !pip3 install -U scikit-learn


# ### Package imports

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import set_config
from sklearn.pipeline import Pipeline
from pandas_profiling import ProfileReport
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
import autosklearn.classification
import PipelineProfiler

import plotly.express as px
import plotly.graph_objects as go

from joblib import dump

import shap

import datetime

import logging

import matplotlib.pyplot as plt


# In[ ]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

from sklearn import preprocessing

from sklearn.metrics import silhouette_score
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


# Connect to Google Drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[1]:


data_path="/content/drive/MyDrive/Introduction2DataScience/W2D2_Assignment/w2d2/data/raw/"
model_path = "/content/drive/MyDrive/Introduction2DataScience/W2D2_Assignment/w2d2/models/"
timesstr = str(datetime.datetime.now()).replace(' ', '_')
logging.basicConfig(filename=f"{model_path}explog_{timesstr}.log", level=logging.INFO)


# In[ ]:


pd.set_option('display.max_rows', 25)


# In[ ]:


set_config(display='diagram')


# Please Download the data from [this source](https://drive.google.com/file/d/1uMM8qdQSiHHjIiYPd45EPzXH7sqIiQ9t/view?usp=sharing), and upload it on your introduction2DS/data google drive folder.

# <a id='P1' name="P1"></a>
# ## [Loading Data and Train-Test Split](#P0)
# 

# **Load the csv file as a DataFrame using Pandas**

# In[ ]:


# your code here
df = pd.read_csv(f'{data_path}data-breast-cancer.csv')

df['diagnosis']=df['diagnosis'].map({'M':1, 'B':0})


# In[ ]:


df = df.drop(["Unnamed: 32", "id"], axis=1) 


# In[ ]:


test_size = 0.2
random_state = 42


# In[ ]:


Y = df['diagnosis']
X = df.drop('diagnosis', axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)


# In[ ]:


logging.info(f'train test split with test_size={test_size} and random state={random_state}')


# In[ ]:


total_time = 100
per_run_time_limit = 30


# In[ ]:


automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=total_time,
    per_run_time_limit=per_run_time_limit,
)
automl.fit(x_train, y_train)


# In[ ]:


# profiler_data= PipelineProfiler.import_autosklearn(automl)
# PipelineProfiler.plot_pipeline_matrix(profiler_data)


# _Your Comments here_

# In[ ]:


logging.info(f'Ran autosklearn regressor for a total time of {total_time} seconds, with a maximum of {per_run_time_limit} seconds per model run')


# In[ ]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[ ]:


logging.info(f'Saved regressor model at {model_path}model{timesstr}.pkl ')


# In[ ]:


logging.info(f'autosklearn model statistics:')
logging.info(automl.sprint_statistics())


# <a id='P2' name="P2"></a>
# ## [Model Evaluation and Explainability](#P0)

# In[ ]:


y_pred = automl.predict(x_test)


# In[ ]:


logging.info(f"Accuracy is {accuracy_score(y_test, y_pred)}, \n F1 score is {f1_score(y_test, y_pred)}")


# #### Model Explainability

# In[ ]:


explainer = shap.KernelExplainer(model = automl.predict, data = x_test.iloc[:50, :], link = "identity")


# In[ ]:


# Set the index of the specific example to explain
x_idx = 0
shap_value_single = explainer.shap_values(X = x_test.iloc[x_idx:x_idx+1,:], nsamples = 100)
x_test.iloc[x_idx:x_idx+1,:]
# print the JS visualization code to the notebook
shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = x_test.iloc[x_idx:x_idx+1,:], 
                show=False,
                matplotlib=True
                )
plt.savefig(f"{model_path}shap_example_{timesstr}.png")
logging.info(f"Shapley example saved as {model_path}shap_example_{timesstr}.png")


# In[ ]:


shap_values = explainer.shap_values(X = x_test.iloc[0:50,:], nsamples = 100)


# In[ ]:


# print the JS visualization code to the notebook
shap.initjs()
fig = shap.summary_plot(shap_values = shap_values,
                  features = x_test.iloc[0:50,:],
                  show=False)
plt.savefig(f"{model_path}shap_summary_{timesstr}.png")
logging.info(f"Shapley summary saved as {model_path}shap_summary_{timesstr}.png")


# In[ ]:





# --------------
# # End of This Notebook
