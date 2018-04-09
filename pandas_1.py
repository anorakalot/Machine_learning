import pandas as pd

#series are a single column
city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])


cities = pd.DataFrame({"City name":city_names, "Population":population})


#print(cities.describe())
#print(cities.head())
#print()

#if reading from a csv
california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe.describe()

#print(california_housing_dataframe.head())

#doesnt work
#california_housing_dataframe.hist("housing_median_age")

#print()

#accessing dataframe objects
#print (cities['City name'][0])


cities['Square area'] = pd.Series([45.123,41.123,9128.21])

print (cities.head())
