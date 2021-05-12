# Import function to create training and test set splits
from sklearn.model_selection import train_test_split
# Import function to automatically create polynomial features! 
from sklearn.preprocessing import PolynomialFeatures
# Import Linear Regression and a regularized regression function
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
# Finally, import function to make a machine learning pipeline
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfz = pd.read_csv('data_parsed.csv')
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


dfz.to_csv('lima.csv')

#dfz0 = pd.read_csv('data0.csv')
#dfz1 = pd.read_csv('data1.csv')
#dfz2 = pd.read_csv('data2.csv')

# Alpha (regularization strength) of LASSO regression
lasso_eps = 0.0001 #0.1
lasso_nalpha=100    # 1
lasso_iter=100000  # 100000
# Min and max degree of polynomials features to consider
degree_min = 2
degree_max = 4
# Test/train split
train, test = train_test_split(dfz, test_size = 0.20, shuffle = False)

print (train.to_numpy())
print('------------------')
print (test.to_numpy)

#X_train, X_test, y_train, y_test = train_test_split(df['X'], df['y'],test_size=test_set_fraction)
# Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
print('#######################')
for degree in range(degree_min,degree_max):
    #model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha))#,max_iter=lasso_iter,normalize=True,cv=5))
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())#,max_iter=lasso_iter,normalize=True,cv=5))
    #print(train[['date','lat','lon','zone']].to_numpy())
    #print('-----------------------')
    #print(train['deaths'].to_numpy())
    #print('-----------------------')
    print(train)
    print(test)
    print(train[['date']].to_numpy())
    model.fit(train[['date']].to_numpy(),train['deaths'].to_numpy())
    i=0
    for index,row in test.iterrows():
        #print("++++++++++++++++++++++++++++++++\n")
        if (i>10):
            break
        #print("Row is: \n", row, "\n****")
        temp = np.array([[row['date']]])
        temp.reshape(1, -1)
        #print(temp)
        #print(model.predict(temp))
        #print(row['deaths'])
        i+=1

    all_pred =   np.array(model.predict(dfz[['date']].to_numpy()))  
    train_pred = np.array(model.predict(train[['date']].to_numpy())) 
    test_pred = np.array(model.predict(test[['date']].to_numpy()))
    #print('##########################################3')
    
    RMSE=np.sqrt(np.sum(np.square(test_pred-test['deaths'].to_numpy())))
    print("test score es: ",RMSE)
    test_score = model.score(test[['date']].to_numpy(),test['deaths'].to_numpy())
    print("Test score: ", test_score)
    #plt.plot(test['date'].to_numpy(), test_pred)
    #plt.plot(train['date'].to_numpy(), train_pred)
    plt.plot(dfz['date'].to_numpy(), all_pred)
    

plt.plot(dfz['date'].to_numpy(), dfz['deaths'].to_numpy(), "*", color="blue")
#plt.plot(test['date'].to_numpy(), test['deaths'].to_numpy(), "*", color="blue")

plt.show()
