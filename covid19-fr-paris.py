import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# Clean hospital data
hospital_data = pd.read_csv('donnees-hospitalieres-covid19/donnees-hospitalieres-covid19-2020-05-01-19h00.csv', sep=';')
paris_data_dep = hospital_data.query('dep=="75" & sexe=="0"').sort_values(by='jour')
paris_data_dep['jour'] = paris_data_dep['jour'].str.slice(start=5)
paris_data_cleaned = paris_data_dep[paris_data_dep['jour'] <= '04-07']

# Clean Paris trafic data
paris_trafic = pd.read_csv('paris_trafic_march.csv')
paris_trafic['jour'] = paris_trafic['day'].str.slice(start=5)
paris_trafic_cleaned = paris_trafic[paris_trafic['jour'] >= '03-11']

# Join data
paris_data_cleaned['key'] = range(1, len(paris_data_cleaned) + 1)
paris_trafic_cleaned['key'] = range(1, len(paris_trafic_cleaned) + 1)
merged = paris_trafic_cleaned.set_index('key').join(paris_data_cleaned.set_index('key'), lsuffix='_tf', rsuffix='_hp')
merged = merged[['jour_tf', 'hosp', 'q']]

# create training and testing datasets
x = merged[['q']]
y = merged.hosp
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=35)
print('Train dataset:', X_train.shape, y_train.shape)
print(' Test dataset:', X_test.shape, y_test.shape)

# fit a model
lin_reg = LinearRegression()
#model1 = lin_reg.fit(X_train, y_train)

i = 1
max = 1000000000000
while(i <= max):
    print(str(i) + " / " + str(max))
    i = i + 1
