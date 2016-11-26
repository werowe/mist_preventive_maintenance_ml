# Spark ML and Hydrosphere Mist Example: Preventive Maintenance
by
Walker Rowe
werowe@walkerrowe.com

## Document reference: 
1. [Hydrosphere Mist Github](https://github.com/Hydrospheredata/mist)
2. [Hydrosphere Mist product page](http://hydrosphere.io/mist/)

 
## Table of Contents

 
1. [Spark ML and Hydrosphere Mist Example: Preventive Maintenance](#1)
2. [Business Assessment: Use Case Background](#2)
3. [Vehicle Fleets and Analytics](#3)	 
4. [Brake Failure Prediction](#4)	 
5. [Brake Pad Maintenance](#5)	 
6. [Ingest data](#6)	 
7. [Prepare data](#7)	 
8. [Train the model](#8)
9. [Test the model](#9)
10. [Expose the Model as a Web Service](#10)
11. [Serve the Model](#11)
12. [Complete Code](#12)
 

## <a name="1"></a>Spark ML and Hydrosphere Mist Example: Preventive Maintenance	

Here we provide an example in Python of how to use Hydrosphere Mist with Spark ML (machine learning library).  Mist lets us expose a model as a web service. 

We take this example from the field of preventive maintenance (PM) as explained below.  Below we discuss the code in depth.  But first we give a use case for why this is needed.


## <a name="2"></a>Business Assessment: Use Case Background
PM was one of the early adopters of big data analytics and machine learning and IoT (Internet of Things) because it is so simple to conceive and implement for that use case.  Calculating when a machine needs maintenance is a problem that fits neatly into a predictive algorithm. This is because machine wear is a function of time and usage.  


## <a name="3"></a>Vehicle Fleets and Analytics
IoT-equipped trucks send data from vehicles using a cellular or satellite signal either as a stream or in bursts.  With IoT, trucks are fit with sensors and GPS trackers that measure heat, vibration, distance travelled, speed, etc.  These sensors are attached to the engine, brakes, transmission, refrigerated trailer, etc.


Companies gather and study this data to operate their vehicles in the safest and lowest cost manner possible.  For example, sensors on the engine can tell whether the engine has a problem.  It is the goal of PM to fix a device before it breaks as waiting until it breaks is expensive as the engine, brake assembly, or drive train can be destroyed and the vehicle taken out of service for a longer period of time than if it is properly maintained


## <a name="4"></a>Brake Failure Prediction
A heavy truck with 18 wheels has a unique preventive maintenance problem to solve, and that is knowing when to change brakes.  Trucks needs to know when to replace their brakes so that they do not have an accident or destroy the brake rotor, which is the metal part of the assembly.  If they wait too long the brake pad will destroy the rotor as metal rubs up against metal.   


The driver cannot be expected to check every brake every time they stop.  And if the company just changes brakes based on some preset schedule then they are wasting money, because they might be changing them too often. So it is preferred to write some mathematical or statistical model to predict when brakes should be changed.  


## <a name="5"></a>Brake Pad Maintenance
Brake pads are metal shavings held together by a resin. The brake applies pressure to the pad to force it down on the rotor, which is a metal disk connected to a truck’s axles.  The pad is designed to wear out over time.  It has to be softer than the rotor, so that it does not damage the rotor.   When the brake pad wears down, heat will go up because there is more friction.  And the further a vehicle has been driven the more its brakes will have worn down.


We contacted an engineer from Volvo and he verified that this model would work as a teaching exercise as it seems reasonable to correlate heat and distance driven with wear.  To get a more accurate model we would have to use something like data from the [IDA Industrial Challenge](https://ida2016.blogs.dsv.su.se/?page_id=1387), which was a competition made by Scana trucking company.


There are lots of factors that impact brake wear.  For example, brakes will wear out faster for vehicles that drive down steep hills.   


We do not have any actual sample data.  So we generated some sample date using this rough model:


`z = wear_rate = (0.003 * heat) + (0.004 * kilometers)-78`


This shows whether the brakes are worn out given the kilometers driven and the maximum heat generated during gathering the sample.


We plug that value into the logistic probability function:


`pr = 1 / (1 + e**-z)`




The binary logistic model generates a binary output, which we will call worn. So if pr > 50% then worn = 1. Otherwise worn = 0. If worn = 1 then it is time to change brake pads.



## <a name="6"></a>Ingest Data
  
For this tutorial, we write two Python programs.  The code for both is located at the bottom of this page.

1. ***brakeTrain.py** to ingest and prepare the data, train the model, and calculate its accuracy.  
2. **brakePredict.py** uses Mist to expose the model as a web service to return a prediction as to whether the brake is worn. 

First we look at brakeTrain.py.  We copy that code into pyspark and run it there.  The pyspark interactive Spark shell gives us a SparkContext without having to code that.  

The sample data is [here](https://raw.githubusercontent.com/werowe/mist_preventive_maintenance_ml/master/brakedata.csv).  Below is the first line.

<table>
<tr>
<td>worn</td><td>km</td><td>heat</td><td>z</td><td>pr</td>
</tr>

<tr>
<td>1</td><td>20,000</td><td>240</td><td>2.72</td><td>0.938197</td>
</tr>

</table>

We read this data into a Pandas data frame and then select only the first three columns: whether the brake is worn, kilometers, brake rotor heat.

```
df = pd.read_csv('/home/walker/hydrosphere/brakedata.csv', sep= ',')
 
brakeData =  df.ix[:,0:3]
```

## <a name="7"></a>Prepare Data
The Spark ML LogisticRegressionWithLBFGS algorithm requires that we put the data into an iterable object of Labels and Points.  So we have an array of LabeledPoint objects.  The Label is the result of logistic regression.  In this case it indicates whether the brake is worn (1) or not (0).  The Points are the kilometers (km) and temperature (heat).


```
a = [] 

def parsePoint(w,k,h):
    return LabeledPoint(worn, [km, heat])


for row in brakeData.itertuples():
	worn = getattr(row, 'worn')
	km = locale.atof(getattr(row, 'km'))
	heat = getattr(row,'heat')
	lp = parsePoint (worn, km, heat)
	a.append(lp)
```


## <a name="8"></a>Train the Model
Now we train the model by passing that array into LogisticRegressionWithLBFGS.train. Then we save the model to disk so that the web service can use it.

Once the model lrm is created, we can call the lrm.predict() method.

```
lrm = LogisticRegressionWithLBFGS.train(sc.parallelize(a))
lrm.save(sc, "/tmp/brakeModel")
```


## <a name="9"></a>Test the Model
To test the model we take the training data and then run the prediction over each data point in the sample data.  We then count how many correct predictions there are and divide that by the sample size.  That calculates the model accuracy.

```
p = sc.parallelize(a)

valuesAndPreds = p.map(lambda p: (p.label, lrm.predict(p.features)))


accurate = 1 - valuesAndPreds.map(lambda (v, p): math.fabs(v-p)).reduce(lambda x, y: x + y) / valuesAndPreds.count()
```
 

## <a name="10"></a>Expose the Model as a Web Service

Finally we expose the model as a web service.  That code is in the second program, brakePredict.py.

The key pieces of the code are to import Mist and then create a class derived from MistJob.  Then implement the method do_stuff().  Then read the parameters and return the results with the return statement.  There is no need instantiate the Predict object.  Mist will do that.

```
from mist.mist_job import MistJob

class Predict(MistJob):
 
    def do_stuff(self, parameters):
        val = parameters.values()
       
        return ("brake is worn=", worn)

```



## <a name="11"></a>Serve the Model
Here is how to call the prediction web service from an external application.  We use CURL to run brakePredict.py. 

First we need to install and configure Hydropshere Mist.


**Configure and Run Mist**  
You can run Mist locally or as a Docker image.


Download and compile Mist:
```
git clone https://github.com/hydrospheredata/mist.git
cd mist
sbt -DsparkVersion=2.0.0 assembly 
```

Create the Mist route by editing:

`vi config/router.conf`

The fields are:

1. **preventineMaintance** is the URL localhost:2004/api/preventineMaintance 
2. **path** is the location of the Python code
3. **className** is the class in the Python program that implements Mist
4. **namespace** is the SparkContext.  It can be any name.


```
preventineMaintance = {
    path = "/home/walker/hydrosphere/brakePredict.py" 
    className = "Predict"
  namespace = "production" 
}


```

IIn config/router.conf put the full path to that in default.conf:

`mist.http.router-config-path = "/home/walker/mist/mist/configs/router.conf"`

Now start Mist changing the version number to match what you have installed:

`./mist start master --config /home/walker/mist/mist/configs/default.conf --jar /home/walker/mist/mist/target/scala-2.11/mist-assembly-0.5.0.jar
`

Run the web service:

`curl --header "Content-Type: application/json" -X POST http://127.0.0.1:2004/api/preventineMaintance --data '{"heatKM":[200,1000]}'`

The port number must agree with the mist.http.port = 2004 in default.conf.
 
Then it responds with the predicted value via the return ("brake is worn=", worn) statement. It loads the training model and then uses lrm.predict([km,heat]) to make the prediction.


## <a name="12"></a>Complete Code

Here is the code:


*BrakeTrain.py*

```

import pandas as pd
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
import locale
import math
locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 

df = pd.read_csv('/home/walker/hydrosphere/brakedata.csv', sep= ',')
 
brakeData =  df.ix[:,0:3]

a = [] 


def parsePoint(w,k,h):
    return LabeledPoint(worn, [km, heat])

for row in brakeData.itertuples():
	worn = getattr(row, 'worn')
	km = locale.atof(getattr(row, 'km'))
	heat = getattr(row,'heat')
	lp = parsePoint (worn, km, heat)
	a.append(lp)


lrm = LogisticRegressionWithLBFGS.train(sc.parallelize(a))

lrm.save(sc, "/tmp/brakeModel")

p = sc.parallelize(a)

valuesAndPreds = p.map(lambda p: (p.label, lrm.predict(p.features)))
accurate = 1 - valuesAndPreds.map(lambda (v, p): math.fabs(v-p)).reduce(lambda x, y: x + y) / valuesAndPreds.count()

```
*BrakePredict.py*

```

from mist.mist_job import MistJob
import os
import numpy as np
import pandas as pd
import math
from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel

class Predict(MistJob):
 
    def do_stuff(self, parameters):
        val = parameters.values()
        list = val.head()
        size = list.size()
        pylist = []
        count = 0
        while count < size:
            pylist.append(list.head())
            count = count + 1
            list = list.tail()


        heat = pylist[0]
        km = pylist[1]
        lrm = LogisticRegressionModel.load(self.context, "/tmp/brakeModel")
        worn = lrm.predict([km,heat])
        return ("brake is worn=", worn)

```



 


