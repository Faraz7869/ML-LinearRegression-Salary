import pandas
import numpy
import joblib

from sklearn.linear_model import LinearRegression

ds=pandas.read_csv("Salary_Data.csv")

X = ds["YearsExperience"].values.reshape(30,1)
Y = ds["Salary"]

mind = LinearRegression()
mind.fit(X ,Y )

print("***************************WELCOME***********************")
print(" ")
print(" ")
output = int(input("Enter your exprerience : "))

ans=mind.predict([[output]])
print(" ")
print("|||||||||||||YOUR ESTIMATED SALARY ||||||||||||||")
print(ans)
print(" ")

joblib.dump( mind ,"SalaryModel.pkl")
