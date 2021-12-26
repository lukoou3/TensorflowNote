import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

def gbt():
    array = np.loadtxt("data/covtype_scaler.csv", delimiter=",", skiprows=1)
    x = array[:, :-1]
    y = array[:, -1]
    x_train,x_test,y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

    gbdt = GradientBoostingClassifier(max_depth=5)

    gbdt.fit(x_train,y_train)

    print("train", gbdt.score(x_train,y_train))
    print("test", gbdt.score(x_test, y_test))



if __name__ == '__main__':
    gbt()