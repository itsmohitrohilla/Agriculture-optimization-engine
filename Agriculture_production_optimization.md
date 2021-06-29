# Agriculture Production Optimization Engine

Project Details :
    
1. This Project is intended on the precision farming
2. we have to optimize productivity.
3. By understanding requirement of climate, soil and conditios for crop.


Task:

Build a predictive model as to suggest the most suitable crops to grow based on the available
climate and soil condition.


Goal:

Achieve Precision farming by optimizing the agriculture production.


```python
#importing Required Libraries which we will use in the project.
```


```python
# for maulplation
import numpy as np
import pandas as pd

#for visulization 
import seaborn as sns
import matplotlib.pyplot as plt

#for interactivity
from ipywidgets import interact
 
```


```python
#read the data set(.csv file)
data = pd.read_csv("data.csv")
```


```python
#print top 5 values of data set
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>P</th>
      <th>K</th>
      <th>temperature</th>
      <th>humidity</th>
      <th>ph</th>
      <th>rainfall</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>42</td>
      <td>43</td>
      <td>20.879744</td>
      <td>82.002744</td>
      <td>6.502985</td>
      <td>202.935536</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>58</td>
      <td>41</td>
      <td>21.770462</td>
      <td>80.319644</td>
      <td>7.038096</td>
      <td>226.655537</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>55</td>
      <td>44</td>
      <td>23.004459</td>
      <td>82.320763</td>
      <td>7.840207</td>
      <td>263.964248</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74</td>
      <td>35</td>
      <td>40</td>
      <td>26.491096</td>
      <td>80.158363</td>
      <td>6.980401</td>
      <td>242.864034</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78</td>
      <td>42</td>
      <td>42</td>
      <td>20.130175</td>
      <td>81.604873</td>
      <td>7.628473</td>
      <td>262.717340</td>
      <td>rice</td>
    </tr>
  </tbody>
</table>
</div>




```python
# checking the shape of the data set(number of rows and number of columns)
data.shape
```




    (2200, 8)




```python
#checking number of null value in the data set.
data.isnull().sum()
```




    N              0
    P              0
    K              0
    temperature    0
    humidity       0
    ph             0
    rainfall       0
    label          0
    dtype: int64




```python
#which crops data is present in the data set and also number of record of that particular data.
data["label"].value_counts()
```




    lentil         100
    grapes         100
    apple          100
    maize          100
    pomegranate    100
    mango          100
    mothbeans      100
    muskmelon      100
    kidneybeans    100
    cotton         100
    orange         100
    papaya         100
    coconut        100
    chickpea       100
    banana         100
    watermelon     100
    blackgram      100
    jute           100
    pigeonpeas     100
    coffee         100
    rice           100
    mungbean       100
    Name: label, dtype: int64




```python
# to check the amount or on an average of minerals are for required and responsible 
#for the crops growth.

print("Average Requirement of minerals for the crop:\n")

print("Nitrogen:",data['N'].mean())
print("Phosphorous: ",data['P'].mean())
print("Postassium: ",data['K'].mean())
print("Temperature: ",data['temperature'].mean())
print("Humidity: ", data['humidity'].mean())
print("PH value: ", data['ph'].mean())
print("RainFall: ", data['rainfall'].mean())
```

    Average Requirement of minerals for the crop:
    
    Nitrogen: 50.551818181818184
    Phosphorous:  53.36272727272727
    Postassium:  48.14909090909091
    Temperature:  25.616243851779544
    Humidity:  71.48177921778637
    PH value:  6.469480065256364
    RainFall:  103.46365541576817



```python
#minumum or maximum and Average requirement of minerals and temperature and humidity etc.
#for the crop. 

#We will use ipywidgets to interect with python code and will make a function to check the 
#required minerals for the specfic crop.
```


```python
@interact

def Requirement(Crops = list(data['label'].value_counts().index)):
    x = data[data['label'] == Crops]
    print("-----Nitrogen-----")
    print("Minimum : ",x['N'].min())
    print("Average : ",x['N'].mean())
    print("Maximum : ",x['N'].max())
    print("------------------")
    
    print("-----Phosphorous-----")
    print("Minimum : ",x['P'].min())
    print("Average : ",x['P'].mean())
    print("Maximum : ",x['P'].max())
    print("------------------")
    
    print("-----Postassium-----")
    print("Minimum : ",x['K'].min())
    print("Average : ",x['K'].mean())
    print("Maximum : ",x['K'].max())
    print("------------------")
    
    print("-----Temperature-----")
    print("Minimum : ",x['temperature'].min())
    print("Average : ",x['temperature'].mean())
    print("Maximum : ",x['temperature'].max())
    print("------------------")
    
    print("-----Humidity-----")
    print("Minimum : ",x['humidity'].min())
    print("Average : ",x['humidity'].mean())
    print("Maximum : ",x['humidity'].max())
    print("------------------")
    
    print("-----Ph Value-----")
    print("Minimum : ",x['ph'].min())
    print("Average : ",x['ph'].mean())
    print("Maximum : ",x['ph'].max())
    print("------------------")
    
    print("-----Rainfall-----")
    print("Minimum : ",x['rainfall'].min())
    print("Average : ",x['rainfall'].mean())
    print("Maximum : ",x['rainfall'].max())
    print("------------------")
    
    
```


    interactive(children=(Dropdown(description='Crops', options=('lentil', 'grapes', 'apple', 'maize', 'pomegranat…



```python
#Those crop requirement is mostly unusual like any particular requirement 
#which a crop want more or less.
print("**High ratio requirments crops as per individual Requirement")
print("--------------------------------------------------------------------------")
print("Crops which required N more than average: ", data[data['N']>120]['label'].unique())
print("Crops which required P more than average: ", data[data['P']>100]['label'].unique())
print("Crops which required K more than average: ", data[data['K']>200]['label'].unique())
print("Crops which required rainfall more than average: ", data[data['rainfall']>200]['label'].unique())
print("Crops which required temperature more than average: ", data[data['temperature']>40]['label'].unique())
print("Crops which required PH more than average:", data[data['ph']>9]['label'].unique())
print("--------------------------------------------------------------------------")
print("**Low ratio requirments crops as per individual Requirement")
print("--------------------------------------------------------------------------")
print("Crops which required temperature less than average: ", data[data['temperature']<10]['label'].unique())
print("Crops which required PH less than average:", data[data['ph']<4]['label'].unique())



```

    **High ratio requirments crops as per individual Requirement
    --------------------------------------------------------------------------
    Crops which required N more than average:  ['cotton']
    Crops which required P more than average:  ['grapes' 'apple']
    Crops which required K more than average:  ['grapes' 'apple']
    Crops which required rainfall more than average:  ['rice' 'papaya' 'coconut']
    Crops which required temperature more than average:  ['grapes' 'papaya']
    Crops which required PH more than average: ['mothbeans']
    --------------------------------------------------------------------------
    **Low ratio requirments crops as per individual Requirement
    --------------------------------------------------------------------------
    Crops which required temperature less than average:  ['grapes']
    Crops which required PH less than average: ['mothbeans']



```python
# TO check the seasonal crops as per the temperature and humidity requirement of the crops:

print("Summer Crops:")
print(data[(data['temperature']<30) & (data["humidity"] > 50)]["label"].unique())
print("--------------------------------------------------------------------------")
print("Winter Crops:")
print(data[(data['temperature']<20) & (data["humidity"] > 30)]["label"].unique())
print("--------------------------------------------------------------------------")
print("Rainy Crops:")
print(data[(data['rainfall']<200) & (data["humidity"] > 30)]["label"].unique())

```

    Summer Crops:
    ['rice' 'maize' 'pigeonpeas' 'mothbeans' 'mungbean' 'blackgram' 'lentil'
     'pomegranate' 'banana' 'mango' 'grapes' 'watermelon' 'muskmelon' 'apple'
     'orange' 'papaya' 'coconut' 'cotton' 'jute' 'coffee']
    --------------------------------------------------------------------------
    Winter Crops:
    ['maize' 'pigeonpeas' 'lentil' 'pomegranate' 'grapes' 'orange']
    --------------------------------------------------------------------------
    Rainy Crops:
    ['rice' 'maize' 'pigeonpeas' 'mothbeans' 'mungbean' 'blackgram' 'lentil'
     'pomegranate' 'banana' 'mango' 'grapes' 'watermelon' 'muskmelon' 'apple'
     'orange' 'papaya' 'coconut' 'cotton' 'jute' 'coffee']



```python
#machine Learning Part
```


```python
#importing Sklearn and we will use KMeans clustring learning
from sklearn.cluster import KMeans

#removing the labels columns
x = data.drop(['label'],axis=1)

#select all the values of data
x = x.values

#check the shape of the data set
x.shape

```




    (2200, 7)




```python
#lets determine the number of cluster in the data set

plt.rcParams['figure.figsize'] =(10,4)

wcss =[]
for i in range(1,11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state =0)
    km.fit(x)
    wcss.append(km.inertia_)
    
#lets plot the result in the graph
plt.plot(range(1,11),wcss)
plt.title("The ELBOW METHOD", fontsize = 20)
plt.xlabel("No of Clusters: ")
plt.ylabel('wcss')
plt.show()
    
```


    
![png](output_17_0.png)
    



```python
#kmeans cluster algorithm implementation
km = KMeans(n_clusters = 4,init = "k-means++", max_iter=300, n_init=10, random_state = 0)
y_means = km.fit_predict(x)

#finding the result
a = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a],axis = 1)
z = z.rename(columns = {0:'cluster'})

#checking cluster of each crop
print("First cluster = ",z[z['cluster']==0]['label'].unique())

print("------------------------------------------------------")
print("Second cluster = ",z[z['cluster']==1]['label'].unique())

print("------------------------------------------------------")
print("Third cluster = ",z[z['cluster']==2]['label'].unique())

print("------------------------------------------------------")
print("forth cluster = ",z[z['cluster']==3]['label'].unique())
```

    First cluster =  ['maize' 'chickpea' 'kidneybeans' 'pigeonpeas' 'mothbeans' 'mungbean'
     'blackgram' 'lentil' 'pomegranate' 'mango' 'orange' 'papaya' 'coconut']
    ------------------------------------------------------
    Second cluster =  ['maize' 'banana' 'watermelon' 'muskmelon' 'papaya' 'cotton' 'coffee']
    ------------------------------------------------------
    Third cluster =  ['grapes' 'apple']
    ------------------------------------------------------
    forth cluster =  ['rice' 'pigeonpeas' 'papaya' 'coconut' 'jute' 'coffee']



```python
# we will use logistic regession for this data set.
```


```python
#lets split teh data for the prdictive modelling

y = data['label']
x = data.drop(['label'], axis = 1)

print(x.shape)
print(y.shape)
```

    (2200, 7)
    (2200,)



```python
#splinting the data for train and test for prediction 
from sklearn.model_selection import train_test_split

x_train , x_test ,y_train ,y_test = train_test_split(x,y, test_size = 0.2, random_state=0)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(x_test.shape)
```

    (1760, 7)
    (440, 7)
    (1760,)
    (440, 7)



```python
#logistic model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```

    /Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages/sklearn/linear_model/_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



```python
from sklearn.metrics import confusion_matrix
#printing the confusion Matrix
plt.rcParams['figure.figsize']=(10,10)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot= True,cmap = 'Wistia')
plt.title("confusion matrix for logistic regression", fontsize= 15)
plt.show()
```


    
![png](output_23_0.png)
    



```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>N</th>
      <th>P</th>
      <th>K</th>
      <th>temperature</th>
      <th>humidity</th>
      <th>ph</th>
      <th>rainfall</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>90</td>
      <td>42</td>
      <td>43</td>
      <td>20.879744</td>
      <td>82.002744</td>
      <td>6.502985</td>
      <td>202.935536</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85</td>
      <td>58</td>
      <td>41</td>
      <td>21.770462</td>
      <td>80.319644</td>
      <td>7.038096</td>
      <td>226.655537</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>55</td>
      <td>44</td>
      <td>23.004459</td>
      <td>82.320763</td>
      <td>7.840207</td>
      <td>263.964248</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74</td>
      <td>35</td>
      <td>40</td>
      <td>26.491096</td>
      <td>80.158363</td>
      <td>6.980401</td>
      <td>242.864034</td>
      <td>rice</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78</td>
      <td>42</td>
      <td>42</td>
      <td>20.130175</td>
      <td>81.604873</td>
      <td>7.628473</td>
      <td>262.717340</td>
      <td>rice</td>
    </tr>
  </tbody>
</table>
</div>




```python
#prediction of a crop
prediction = model.predict((np.array([[90,40,40,20,80,7,200]])))
print("Prediction", prediction)
```

    Prediction ['rice']



```python
#our model is accurate
```


```python
#finish 
```
