import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df=pd.read_excel('Folds5x2_pp.xlsx')
print(df.dropna().sum())

X = df.iloc[:,:-1].values
y=df.iloc[:,-1].values

sc = StandardScaler()
X_train,X_test,y_train,y_test = train_test_split(X,y)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation='relu',))
ann.add(tf.keras.layers.Dense(units=6,activation='relu',))
ann.add(tf.keras.layers.Dense(units=1))
ann.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
ann.fit(X_train,y_train,epochs=100)
y_pred=ann.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_pred),1)),1))
print(accuracy_score(y_pred,y_test))