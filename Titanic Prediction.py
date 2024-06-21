#!/usr/bin/env python

import pandas as pd
d = {'a':1,'b':2,"c":5,"f":4}
pd.Series(d)

d = {'a':1,'b':2,"c":5,"f":4}
ser = pd.Series(data = d,index = ['a','b','c','d'])
ser

#indexing..
cities = ['kolkata','mumbai','tornato','lisbon']
populations = [14.56,2.61, 2.93,0.51]
city_series = pd.Series(populations,index = cities)
city_series.index

import pandas as pd
import numpy as np
dates = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8,4), index = dates, columns = ['A','B','C','D'])
df

s = df['A']
s[dates[5]]

df
df[['B','A']] = df[['A','B']]
df
df[['A','B']]
df[['A','B']]

#swap,,,col in using row values..
df.loc[:]

#rename..
import pandas as pd
d = {'col1': [1,2,3,4,5], 'col2':[5,4,3,2,1],'col3':[9,8,7,6,5]}
df = pd.DataFrame(data=d)
print("Original Dataframes")
print(df)
df.columns = ['Column1', 'Column2', 'Column3']
df = df.rename(columns={'col1': 'Column1','col2':'Column2', 'col3':'Column3'})
print("New dataframe after renaming columns:")
print(df)

#select row..
import pandas as pd
import numpy as np
d = {'col1': [1,2,3,4,5], 'col2':[5,4,3,2,1],'col3':[9,8,7,6,5]}
df = pd.DataFrame(data=d)
print("Original Dataframes")
print(df)
print(df.loc[df['col1']==4])
print(df)

#inter col..
import pandas as pd
d = {'col1': [1,2,3,4,5], 'col2':[5,4,3,2,1],'col3':[9,8,7,6,5]}
df = pd.DataFrame(data=d)
print("Original Dataframes")
print(df)
df = df[['col3','col2','col1']]
print(df)

#add data
import pandas as pd
d = {'col1': [1,2,3,4,5], 'col2':[5,4,3,2,1],'col3':[9,8,7,6,5]}
df = pd.DataFrame(data=d)
print("Original Dataframes")
print(df)
df2 = {'col1':10,'col2':11,'col3':12}
df = df.append(df2, ignore_index=True)
print(df)

#using tab operator,,.
import pandas as pd
d = {'col1': [1,2,3,4,5], 'col2':[5,4,3,2,1],'col3':[9,8,7,6,5]}
df = pd.DataFrame(data=d)
print("Original Dataframes")
print(df)
df.to_csv('new_file.csv', sep= '\t', index = False)
new_df = pd.read_csv('new_file.csv')
print(new_df)
print('wali')

import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv")

df
df.columns
df.head(4)
df.tail(2)
df.dtypes
df.info()

df
df.describe() #gives numerical column.
df['Survived']
df.dtypes == 'object'
df.dtypes[df.dtypes == 'object'] #col name 
df.dtypes[df.dtypes == 'object'].index 
df.dtypes[df.dtypes != 'object'].index 
df[df.dtypes[df.dtypes != 'object'].index].describe()
df.describe()
df.describe(include = 'object')
df.describe(include = 'all')
df.describe()#five points summary..
df.astype
df.astype('object').describe() #gives obj statotical infor categorical.

df
df[0:100]
df[0:100:5]

df
df['new_col'] = "pwskills"

df
df['family'] = df['SibSp'] + df['Parch']

df

pd.Categorical(df['Pclass']) #differ categories and class..
pd.Categorical(df['Cabin'])
df['Cabin'].unique()
df['Cabin'].value_counts()

df
df['Age'] <5
df[df['Age'] < 5]
len(df[df['Age'] < 5])
df[df['Age'] < 5].Name
list(df[df['Age'] < 5].Name)
list(df[df['Age'] > 18].Name)
len(df[df['Age'] > 18].Name)

df['Fare']

df['Fare'].mean()

df['Fare'] < 32.20

df[df['Fare'] < 32.20]

list(df['Fare'] < 32.20)

len(df['Fare'] < 32.20)
df['Fare'] == 0
df[df['Fare'] == 0]
len(df[df['Fare'] == 0])
list(df['Fare'] == 0)
df[df['Fare'] == 0].Name
df[df['Sex']== "male"]
len(df[df['Sex']== "male"])
len(df[df['Sex']== "female"])
df['Sex'].value_counts(normalize = True)
df['Pclass'] == 1
df[df['Pclass'] == 1]
df[df['Survived'] == 1]
len(df[df['Survived'] == 1])
df['Survived'].value_counts(normalize = True)
df['Sex'] == 'Female'
df['Fare'].mean()
df[(df['Sex'] == 'female')  &  (df['Fare']> df['Fare'].mean())]
len(df[(df['Sex'] == 'female')  &  (df['Fare']> df['Fare'].mean())])
len(df[(df['Sex'] == 'male')  &  (df['Fare']> df['Fare'].mean())])


import numpy as np
np.mean(df.Fare)

df.Fare.mean()
max(df.Fare)
df['Fare'] == max(df.Fare)
df[df['Fare'] == max(df.Fare)]
df.Age
(df.Age > 18 ) & (df['Survived'] == 1)
df[(df.Age > 18 ) & (df['Survived'] == 1)]

df[0:100]
df.iloc # iimplicit index 0 to 2 index
df.iloc[0:2]
df

df.loc[0:2]
df.loc[0:2,['Name','Parch']]
df.iloc[0:2, 3:6] #3 sae 6 tkk chaiye. 

df['Name'][2:5]

pd.Series(list(df['Name'][2:5]), index = ['a','b','c'])

s = pd.Series(list(df['Name'][2:5]))
s

"pw" + "skills"

s+s1

s1 = pd.Series(list(df['Name'][5:8]), index = ['a','b','c'])
s1

s

s + s1

df.drop('PassengerId',axis = 1)

df

df.drop('PassengerId', axis = 1, inplace = True)

df

df.drop(1, inplace = True)

df

df.reset_index(drop = True)

df.set_index('Name')
df.set_index('Name', inplace = True)

df

df.loc[Johnston, Miss. Catherine Helen "Carrie"]
df.reset_index(inplace = True)

df

d = {'a':[2,3,4,5],
    'b':[4,5,6,7],
    'c':[2,3,4,5]}

d
pd.DataFrame(d)

df1 = pd.read_csv("customers-100.csv")

df1

df1.shape

df1.describe()

df1.info()

df1

df1.isnull()
df1.isnull().sum()

df

df1

df1.dropna()    #remove null values..

df1
df1.dropna()
df1.dropna(axis = 1)
df1[['Customer Id']].dropna(axis = 1)

df1
df1.fillna("somevalue")
df1.fillna(0)
df1.parent_id.fillna()
data1 = {
    'A':[1,2,None,4,5,None,7,8,9,10],
    'B':[None,11,12,13,None,15,16,None,18,19]
}
df2 = pd.DataFrame(data1)
df2
df2.A
df2.A.fillna(df2['A'].mean())
df2.A.fillna(df2['A'].median())
df2
df2.fillna(0)
df2.fillna('something')
df2.fillna(method = 'ffill') #forward fill-> start seeing rom bottom
df2.fillna(method = 'bfill') # top to .. fil next
df2
df2.duplicated()
df2.duplicated().sum()
df.mean()
df.median()
df.std()
df.cov()
df.Age.describe()
df
df[df.Survived == 1]['Fare'].mean()
df[df.Survived == 0]['Fare'].mean()
df[df.Survived == 0]['Age'].mean()
df.groupby('Survived').mean()
df.groupby('Survived').mean(numeric_only = True)
df.groupby('Survived').median(numeric_only = True)
df.groupby('Survived').sum(numeric_only = True)
df.groupby('Survived').describe()
import numpy as np
df.groupby(['Survived'])['Fare'].agg([min,'max','mean','median','count',np.std,'var'])
df[['Survived','Fare']]
[image.png]()

df.groupby

import pandas as pd
df.groupby(['Sex','Pclass'])['Survived'].sum()
df.groupby(['Sex','Pclass'])['Survived'].sum().to_frame()

df.groupby(['Sex','Pclass'])['Survived'].sum().unstack()

df

a = df.groupby('Pclass').sum(numeric_only = True)

a
a.transpose
df.head()
df.head().T
import pandas as pd
import numpy as np

df
df1 = df[["Name","Sex","Age"]]
df1
df1 = df[["Name","Sex","Age"]][0:5]

df1
df2 = df[["Name","Sex","Age"]][5:10]
df2

pd.concat([df1,df2], axis = 0)

pd.concat([df1,df2], axis = 1)
df2.reset_index(drop = True)

df5 = pd.DataFrame({'a':[1,2,3,45,6],
                    'b':[45,5,7,8,12],
                   'c':[87,8,56,34,67]
}
)

df5

df6 = pd.DataFrame({'a':[1,2,3,5,6],
                    'b':[5,5,7,8,2],
                   'c':[7,8,6,4,7]
}
)

df6

pd.merge(df5,df6, how = 'inner')
pd.merge(df5,df6, how = 'left')
pd.merge(df5,df6, how = 'right')
pd.merge(df5,df6, how = 'outer') #both data frame.
pd.merge(df5,df6, how = 'cross')
pd.merge(df5,df6,how = "left", left_on = "a", right_on = "c")
pd.merge(df5,df6,how = "left", left_on = "a", right_on = "b")

v  = pd.DataFrame({'a':[1,2,3,45,6],
                    'b':[45,5,7,8,12],
                   'c':[87,8,56,34,67]},
                   index = ['x','y','z','t','s']
)

v

v1  = pd.DataFrame({'p':[1,2,3,45,6],
                    'q':[45,5,7,8,12],
                   'r':[87,8,56,34,67]},
                   index = ['x','y','k','i','m']
)

v1

v.join(v1,how = "inner") #common

v.join(v1,how = "outer")
v.join(v1,how = "left")

v.join(v1,how = "cross")

v.join(v1,how = "right")

df

df.Fare #dollar

df['Fare'].apply

df['Fare_inr'] = df['Fare'].apply(lambda x:x*90) #ekk print prr apply krni hai data print kae.

df['Fare_inr']

len(df['Fare_inr'])

df

df['Name_len'] = df['Name'].apply(len)

df['Name_len']

def convert(x):
    return x*90

df['Fare'].apply(convert)

def create_flag():
    if x < 10:
        return "cheap"
    elif x >= 10 and x<20:
        return "medium"
    else:
        return "hig"
 
df['flag_fare'] = df['Fare'].apply(create_flag)

df

v
v.set_index('a', inplace = True)

v
v.reset_index(inplace = True)

v
v.reindex(['a','g','u','p','r'])

v.reindex([0,1,2,3,4])
v

for i in v.iterrows(): #iteration rows wise..
    print(i,"..............")

for i in v1.items():#col wise
    print(i)

def fun_sum(x):
    return x.sum()

v.apply(func_sum, axis = 0)
