import numpy as np
import pandas as pd

import sklearn.datasets

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

brest_cancer_dataset = sklearn.datasets.load_breast_cancer()

class hypermodel:
    is_legit = False

    def __init__(self,test) -> bool:
        self.df = pd.DataFrame(brest_cancer_dataset.data , columns = brest_cancer_dataset.feature_names)
        self.df['Label'] = brest_cancer_dataset.target

        #Distribution of Taregt (Label) variable
        # BENIGN --> 1
        # MALIGNANT --> 0

        self.X = self.df.drop(columns= 'Label' , axis = 1)
        self.Y = self.df['Label']
        X_train , X_test , Y_train , Y_test = train_test_split(self.X , self.Y , test_size = 0.2 , random_state = 2)

        #MODEL TRAINING 
        model = LogisticRegression()
        model.fit(X_train , Y_train)
        X_train_prediction = model.predict(X_train)
        training_data_accuracy = accuracy_score(Y_train , X_train_prediction)
        print("Accuracy Score of the TRAINING DATA : ",training_data_accuracy*100)
        X_test_prediction = model.predict(X_test)
        testing_data_accuracy = accuracy_score(Y_test , X_test_prediction)
        print("Accuracy Score of TESTING DATA : ",testing_data_accuracy*100)

        #PREDICTIVE SYSTEM
        inp_data = [test]
        inp_data_as_nparray = np.asarray(inp_data)
        inp_reshaped_data = inp_data_as_nparray.reshape(1, -1)

        prediction = model.predict(inp_reshaped_data)
        if(prediction == 0):
            print("MALIGNANT CANCER")
            self.is_legit = False
        else:
            print("BENIGN CANCER")
            self.is_legit = True

        pass