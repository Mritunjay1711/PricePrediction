from joblib import dump, load
import numpy as np
model = load("RealEstate.joblib")


print("Enter the 13 attributes of house")
list = []

for i in range(0, 13):
    ele = float(input())
    list.append(ele)

input = np.array([list])
print(f"MEDV for these features {model.predict(input)}")
