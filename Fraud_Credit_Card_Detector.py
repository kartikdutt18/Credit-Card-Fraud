import pandas as pd
import numpy as np
dataset=pd.read_csv('Credit_Card_Applications.csv')
y=dataset.iloc[:,-1].values
x=dataset.iloc[:,:-1].values
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
x=sc.fit_transform(x)
from minisom import MiniSom
som=MiniSom(10,10,input_len=15,sigma=1.0,learning_rate=0.1)
som.random_weights_init(x)
som.train_random(x,128)
from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o','s']
colors=['r','g']
for i,j in enumerate(x):
    w=som.winner(j)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None')
show()
mappings=som.win_map(x)
frauds=np.concatenate((mappings[(8,1)],mappings[(6,8)]),axis=0)
frauds=sc.inverse_transform(frauds)
data2=dataset.iloc[:,1:].values
fr1=np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        fr1[i]=1
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
sc1.fit_transform(data2)
from keras.models import Sequential
from keras.layers import Dense
clas=Sequential()
clas.add(Dense(units=2,input_dim=15,kernel_initializer='uniform',activation='relu'))
clas.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
clas.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
clas.fit(data2,fr1,batch_size=1,epochs=4)
pred=clas.predict(data2)
pred=np.concatenate((dataset.iloc[:,0:1].values,pred),axis=1)
pred=pred[pred[:,1].argsort()]