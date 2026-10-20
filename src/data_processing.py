import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    data = pd.read_csv(r"data\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    return data

def preprocess_data(data):
    # data (as pandas dataframes) 
    X = data.drop(columns=['Diabetes_binary'])
    y = data['Diabetes_binary']
    
    # metadata 
    print(data.info()) 
    
    # variable information 
    print(data.describe()) 

    #checking data count
    print(y.value_counts())

    #checking missing values
    print(data.isnull().sum())

    #checking for duplicates
    print(data.duplicated().sum())

    #splitting data into training set and test set and proportion of data in each set is maintained using stratify parameter
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify = y)   
    print("Training set size:", x_train.shape)
    print("Test set size:", x_test.shape)

    #initialize scaling feature
    scaler = StandardScaler()

    #calculating mean and standard deviation for scaling
    scaler.fit(x_train)

    #transform training data and scale
    x_train_scaled = scaler.transform(x_train)

    #apply same scaling to test data
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test

