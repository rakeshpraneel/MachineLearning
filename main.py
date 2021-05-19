#It is also known as scikit learn, which is a ML library
import sklearn
from sklearn import linear_model
import numpy as np

#This program is an example for supervised learning model where the model is trained using both features and labels

if __name__ == '__main__':
    x = list(range(1,10)) #Input (feature) which is mtrs
    y = [39.3701, 78.7402, 118.11, 157.48, 196.85, 236.22, 275.591, 314.961, 354.331] #Output (Labels) which is inches

    #This reshaping is done in order as the models get input in these format
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    #print(f'x: {x}')
    #print(f'y: {y}')

    #Spliting the train and test data for the model
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    #initating the model
    model = linear_model.LinearRegression()

    try:
        #Training the model using fit function
        #Here the feature and labels that are allocated for training set is passed
        model.fit(xtrain, ytrain)

        #Calculating the models accuracy rate
        accuracy = model.score(xtest, ytest)
        print("Accuracy of the model is: ", accuracy * 100)

        #Obtaining input from the user & reshaping it to pass into model
        userinput = int(input("Enter the value : "))
        userinput = np.array(userinput).reshape(-1,1)
        #print("Reshaped userinput: ", userinput)

        #Using predict module the userinput is passed inot the model
        predicted_value = float(model.predict(userinput))
        print(round(predicted_value,2))

    except Exception:
        print("Model training failed!!!")
        print(Exception)