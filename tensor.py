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
#TrainingSet = numpy.genfromtxt("dataset/data_parsed.csv", delimiter=",", skip_header=True)
#ValidationSet = numpy.genfromtxt("validation.csv", delimiter=",", skip_header=True)
dfz = pd.read_csv('dataset/data_parsed.csv')
ciudades = dfz.region.unique()
print(ciudades)
print('escoga la ciudad donde quiere hacer el analisis:')
indice1 = 0
for i in range(len(ciudades)):
    print('[',i,'] ',ciudades[i])

indice1 = int(input("ingrese el codigo de la ciudad: "))

Limas = ['LIMA METROPOLITANA','LIMA REGION','LIMA']
indice2= -1

if(ciudades[indice1]=='LIMA'):
    for i in range(len(Limas)):
        print('[',i,'] ',Limas[i])
    indice2 = int(input("ingrese el codigo"))

if (indice1!=-1):
    dfz= dfz[dfz.region_orig==Limas[indice2]]
else:
    dfz= dfz[dfz.region==ciudades[indice1]]


dfz.to_csv('dataset/lima.csv')

# Test/train split
train, test = train_test_split(dfz, test_size = 0.20, shuffle = False)


# split into input (X) and output (Y) variables
X1 = train[['date']].to_numpy() #TrainingSet[:,0:0]
Y1 = train[['deaths']].to_numpy() #TrainingSet[:,5]

X2 = test[['date']].to_numpy() #ValidationSet[:,0:5]
Y2 = test[['deaths']].to_numpy() #ValidationSet[:,5]

X3 = dfz[['date']].to_numpy() #ValidationSet[:,0:5]
Y3 = dfz[['deaths']].to_numpy() #ValidationSet[:,5]


print(X1)
print(Y1)
print("##############3")
print(X2)
print(Y2)


#print(asdf)

# create model
model = Sequential()
#model.add(Dense(100, activation="relu", input_dim=1, kernel_initializer="uniform"))
model.add(Dense(20, activation="relu", input_dim=1, kernel_initializer="uniform"))
model.add(Dense(1, kernel_initializer="uniform"))
#model.add(Dense(1, use_bias =  True, input_shape=(1,)))

# Compile model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X1, Y1, epochs=15000, batch_size=10,  verbose=2)

# Calculate predictions
PredTestSet = model.predict(X1)
PredValSet = model.predict(X2)
PredX3 = model.predict(X3)


# Save predictions
numpy.savetxt("trainresults.csv", PredTestSet, delimiter=",")
numpy.savetxt("valresults.csv", PredValSet, delimiter=",")

#Plot actual vs predition for training set
TestResults = numpy.genfromtxt("trainresults.csv", delimiter=",")
#plt.plot(Y1,TestResults,'ro')
#plt.title('Training Set')
#plt.xlabel('Actual')
#plt.ylabel('Predicted')

#Compute R-Square value for training set
TestR2Value = r2_score(Y1,TestResults)
print("Training Set R-Square=", TestR2Value)

#Plot actual vs predition for validation set
ValResults = numpy.genfromtxt("valresults.csv", delimiter=",")

plt.plot(X3,Y3,'*')
plt.plot(X3,PredX3)
plt.show()
#plt.plot(Y2,ValResults,'ro')

#plt.title('Validation Set')
#plt.xlabel('Actual')
#plt.ylabel('Predicted')
#plt.show()
#Compute R-Square value for validation set
ValR2Value = r2_score(Y2,ValResults)
print("Validation Set R-Square=",ValR2Value)