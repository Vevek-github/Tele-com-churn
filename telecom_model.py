# pip install -U scikit-learn==1.2.2   if code throws errors ,use in terminal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, f1_score
import pickle
import joblib

def input_run(dff:pd.DataFrame):

  # Input
  
  sample_size = len(dff)

  
  # Load the models and encoders using pickle

  best_gb = joblib.load(r"Models\best_gb_telecom")
  simple_scaler  = joblib.load(r"Encoders\simple_scaler")
  mean_encoder = joblib.load(r"Encoders\mean_encoder_model")
  enc = joblib.load(r"Encoders\ohe_area_code")

  #sample_size =20
  # index_set for merge
  #sample_df = dff.sample(n=sample_size, random_state=42)
  #sample_df.index = [x for x in range(sample_size)]
  dff.index = [x for x in range(sample_size)]
  df =dff.copy()
  if 'Unnamed: 0' in df:
    del df['Unnamed: 0']

  if 'churn' in df:
    del df['churn']


  # Data Pre-processing
  feature_values  = enc.transform(df[['area.code']]).toarray()
  feature_labels  = enc.categories_[0]
  new_labels  = []                                                                # One-hot encoding round about
  for i in feature_labels:
    new_labels.append("area.code_"+i)
  df=pd.concat([pd.DataFrame(feature_values,columns=new_labels),df],axis =1)      # Merged one hot encoding  
  if df['day.charge'].dtype!=np.number :
    df['day.charge']=df['day.charge'].astype('float')
  if df['eve.mins'].dtype!=np.number :
    df['eve.mins']=df['eve.mins'].astype('float')
  df=df.dropna()                                               # drop null values
  num_cols = df.select_dtypes(include = np.number).columns
  cat_cols = df.select_dtypes(include ='object').columns


  df["intl.plan"] = df["intl.plan"].map({"no": 0, "yes": 1})   # Binary encoding
  df["intl.plan"] = df["intl.plan"].astype("int64")
  
  
  
  
  df["voice.plan"] = df["voice.plan"].map({"no": 0, "yes": 1})
  df["voice.plan"] = df["voice.plan"].astype("int64")
  
  
  
  #df["churn"] = df["churn"].map({"no": 0, "yes": 1})
  
  #df["churn"] = df["churn"].astype("int64")
  #df = pd.get_dummies(df, columns=['area.code'],dtype = int)   # Label Encoding
  df['state_target']= mean_encoder.transform(df['state'])      # State mean encoding
  columns_to_drop = ["eve.mins",   "day.mins",  "night.mins",  "intl.mins",'voice.plan','state','area.code']
  df = df.drop(columns_to_drop, axis=1)                        # Dropping state varible
  feature_to_scaler =['account.length','voice.messages','day.calls','eve.calls','night.calls','intl.calls']
  df[feature_to_scaler]= simple_scaler.transform(df[feature_to_scaler])   # Simple scalers


  # Re-order the column names 
  feature_names  =  ['account.length', 'voice.messages', 'intl.plan', 'intl.calls',
        'intl.charge', 'day.calls', 'day.charge', 'eve.calls', 'eve.charge',
        'night.calls', 'night.charge', 'customer.calls',
        'area.code_area_code_408', 'area.code_area_code_415',
        'area.code_area_code_510', 'state_target']
  df = df[feature_names]      

  # load the gbm 
  return best_gb.predict(df)

