from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import train_test_split

#%matplotlib inline


#Red data from csv file for training and validation data

dfz = pd.read_csv('dataset/data_parsed.csv')

dfz.to_csv('dataset/lima.csv')

# Test/train split




def mse(y,y_pd): # minimum square error
    return sum([(e[0]-e[1])**2 for e in zip(y,y_pd)])/(2*len(y))

def tester2(x_ds,y_ds,city,X2,X3,Y3):
  color = { None:"yellow","momentum":"red", "adadelta":"blue","adam":"green","adagrad":"purple"}
  for k in ["l1","l2",None]:
    for i in ["mse","mae"]:
      plt.plot(X3,Y3,'*')
      for j in ["adadelta","adagrad","adam"]:
        model = Sequential()
        if k==None:
            model.add(Dense(20, activation="relu", input_dim=1, kernel_initializer="uniform"))
        else:    
            model.add(Dense(20, activation="relu", input_dim=1, kernel_initializer="uniform", kernel_regularizer=k))
        model.add(Dense(1, kernel_initializer="uniform"))
        
        model.compile(loss=i, optimizer=j, metrics=['accuracy'])
        history = model.fit(x_ds, y_ds, epochs=10000, batch_size=10,  verbose=0)
        #PredTestSet = model.predict(X1)
        #PredValSet = model.predict(X2)
        PredX3 = model.predict(X3)
        plt.plot(X3,PredX3, color=color[j],label=j)
        if k!=None:
            print('finished ' + j  + ' ' + i  + ' ' + k)
        else:
            print('finished ' + j  + ' ' + i)
        print(mse(Y3,PredX3))
      if k!=None:
        name = "test4/" + city +"/" + i + "_" + str(k)
      else:
        name = "test4/" + city +"/" + i 
      print(name)
      plt.legend()
      plt.savefig(name)
      plt.clf()

#exec tester
def exec_tester():
    dfz = pd.read_csv('dataset/data_parsed.csv')
    dfz.to_csv('dataset/lima.csv')
    cities = ["UCAYALI", "AREQUIPA"]
    
    for i in cities:
        dfz2= dfz[dfz.region==i]
        train, test = train_test_split(dfz2, test_size = 0.20, shuffle = False)

        X1 = train[['date']].to_numpy() #TrainingSet[:,0:0]
        Y1 = train[['confirmed']].to_numpy() #TrainingSet[:,5]

        X2 = test[['date']].to_numpy() #ValidationSet[:,0:5]
        Y2 = test[['confirmed']].to_numpy() #ValidationSet[:,5]

        X3 = dfz2[['date']].to_numpy() #ValidationSet[:,0:5]
        Y3 = dfz2[['confirmed']].to_numpy() #ValidationSet[:,5]
        tester2(X1,Y1, i, X2,X3,Y3)

exec_tester()
