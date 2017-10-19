import matplotlib.pyplot as plt
import pandas
import numpy
from sklearn import preprocessing
from sklearn import utils
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def main():
    
    #womens breast cancer files
    file = "data/wdbc.csv"

    #data cols 0, 1, id and diagnosis
    keys = ['id','diag']

    #10 features x 3 variations = 30 features

    features = ['radius','texture','perimiter','area','smoothness','compactness','concavity','concpoints','symmetry','fractald']

    variations = ['mean','SE','largest']
    
    #combine features and variations
    allfeatures=[]
    for variation in variations:
        for feature in features:
            allfeatures.append(variation+'_'+feature)
     
    #all column names     
    names = keys+allfeatures
    dataframe = pandas.read_csv(file, names=names)
    
    array=dataframe.values


    y = dataframe['diag'].values
    x = dataframe[allfeatures].values
    

    x = standardizeData(x)
    #plotData(dataframe)
    #printData(data)
    crossValidate(x,y)

def printData(data):
    print("#"*15+"head")
    print(data.head())
    print()
    print("#"*15+"shape")
    print(data.shape)
    print()
    print("#"*15+"dtypes:")
    print(data.dtypes)
    print()
    print("#"*15+"describe")
    print(data.describe())
    print()
    print("#"*15+"correlation")
    print(data.corr())


    
def plotData(data):
    scatter_matrix(data, figsize=(18, 18))
    plt.show()
    

def standardizeData(df):
    rescaledX = (df - df.mean()) / df.std()
    return rescaledX
    
def crossValidate(x, y):
    kfold = KFold(n_splits=10, random_state=7)
    model = LogisticRegression()
    results = cross_val_score(model, x, y, cv=kfold)
    print("Accuracy:{:10.3f} {:10.3f}".format( results.mean()*100.0, results.std()*100.0))


    
    


#########################################################
###############        MAIN         #####################
#########################################################
main()