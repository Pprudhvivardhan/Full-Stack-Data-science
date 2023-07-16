 
import numpy as np
import pandas as pd

df = pd.read_csv('hiring.csv')
df['experience'].fillna(0,inplace=True)
df['test_score'].fillna(df['test_score'].mean(),inplace=True)

x=df.iloc[:,:3]
y=df.iloc[:,-1]

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)
model.coef_
model.intercept_


import pickle
pickle.dump(model,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))
