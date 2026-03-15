import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

dataset = pd.read_csv("diabetes.csv")

x = dataset.drop("Outcome",axis=1)
y = dataset["Outcome"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=45)

model = Lasso(alpha=0.1)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

# convert regression output to classification
y_pred_class = np.where(y_pred>0.5,1,0)



accuracy = accuracy_score(y_test,y_pred_class)

print("Model Accuracy:",accuracy )

# save model

joblib.dump(model,"diabetes_lasso_model.pk1")

print("model saved successfully")