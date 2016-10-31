# Spark ML and Hydrosphere Mist Example: Preventive Maintenance


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

Here we provide an example in Python of how to use Hydrosphere Mist with Spark ML (machine learning library).  We take this example from the field of preventive maintenance (PM) as explained below.


This is a tutorial.  Here is what you will learn:
 
Use Hydrosphere Mist to run a Spark job.
Create and train a logistic regression model using Spark ML and Python.
Write a webservice in Python using Hydrosphere Mist that predicts when brakes need to be replaced on a heavy truck.
Train model and save it.  Then use it for prediction queries.
Evaluate the model for accuracy. 


Below we discuss the code in depth.  But first we give a use case for why this is needed.


## <a name="2"></a>Business Assessment: Use Case Background
PM was one of the early adopters of big data analytics and machine learning and IoT (Internet of Things) because it is so simple to conceive and implement for that use case.  Calculating when a machine needs maintenance is a problem that fits neatly into a predictive algorithm. This is because machine wear is a function of time and usage.  


## <a name="3"></a>Vehicle Fleets and Analytics
IoT-equipped trucks send data from vehicles using a cellular or satellite signal either as a stream or in bursts.  With IoT, trucks are fit with sensors and GPS trackers that measure heat, vibration, distance travelled, speed, etc.  These are attached to the engine, brakes, transmission, refrigerated trailer, etc.


Companies gather and study this data to operate their vehicles in the safest and lowest cost manner possible.  For example, sensors on the engine can tell whether the engine has a problem.  It is the goal of PM to fix a device before it breaks as waiting until it breaks is expensive as the engine, brake assembly, or drive train can be destroyed and the vehicle taken out of service for a longer period of time than if it is properly maintained


## <a name="4"></a>Brake Failure Prediction
A heavy truck with 18 wheels has a unique preventive maintenance problem to solve, and that is knowing when to change brakes.  Trucks needs to know when to replace their brakes so that they do not have an accident or destroy the brake rotor, which is the metal part of the assembly.  If they wait too long the brake pad will destroy the rotor as metal rubs up against metal.   


The driver cannot be expected to check every brake every time they stop.  And if the company just changes brakes based on some preset schedule then they are wasting money, because they might be changing them too often. So it is preferred to write some mathematical or statistical model to predict when brakes should be changed.  


## <a name="5"></a>Brake Pad Maintenance
Brake pads are metal shavings held together by a resin. The brake applies pressure to the pad to force it down on the rotor, which is a metal disk connected to a truck’s axles.  The pad is designed to wear out over time.  It has to be softer than the rotor, so that it does not damage the rotor.   When the brake pad wears down, heat will go up because there is more friction.  And the further a vehicle has been driven the more its brakes will have worn down.


We contacted an engineer from Volvo and he verified that this model would work as a teaching exercise as it seems reasonable to correlate heat and distance driven with wear.  To get a more accurate model we would have to use something like this winning paper from IDA Industrial Challenge, which was a competition made by Scana trucking company.


There are lots of factors that impact brake wear.  For example, brakes will wear out faster for vehicles that drive down steep hills.   


We do not have any actual sample data.  So we generated some sample date using this rough model:


`z = wear_rate = (0.003 * heat) + (0.004 * kilometers)-78`


This shows whether the brakes are worn out given the kilometers driven and the maximum heat generated during gathering the sample.


We plug that value into the logistic probability function:


`pr = 1 / (1 + e**-z)`




The binary logistic model, logit, requires a binary output. So if pr > 50% then worn = 1. Otherwise worn = 0. If worn = 1 then time to change brake pads.

The sample data is [here](https://raw.githubusercontent.com/werowe/mist_preventive_maintenance_ml/master/brakedata.csv).  Below is the first line.
<table>
<tr>
<td>worn</td><td>km</td><td>heat</td><td>z</td><td>pr</td>
</tr>

<tr>
<td>1</td><td>20,000</td><td>240</td><td>2.72</td><td>0.938197</td>
</tr>


</table>




 
## <a name="6"></a>Ingest Data
We expose the data model as a web service for enterprise applications.  


To actually deploy this for a trucking company, a truck fleet manager would use IoT to upload truck data to a web service and call the model to get a prediction. The enterprise application would create a maintenance work order in the preventive maintenance (PM) system when a brake pad needs replacing and notifies the driver.  Then the driver returns to the truck headquarters for repairs or goes to a repair shop.


Obviously we did not write the IoT or interface to the PM system here, since that would be different for different companies and require an IoT cloud, truck, and ERP system.


For this tutorial, we write three Python programs:

1) brakeTrain.py to train the model and calculate its efficiency..
2) brakePredict.py to use that model and return a prediction as to whether the brake is probably worn.
 

## <a name="7"></a>Prepare Data
Download the training data from Github [here](https://raw.githubusercontent.com/werowe/mist_preventive_maintenance_ml/master/brakedata.csv).


Copy the code below into PySpark and run it there.


 
## <a name="8"></a>Train the Model
This job is exposed as a web service by Mist.   

## <a name="9"></a>Test the Model

## <a name="10"></a>Expose the Model as a Web Service

## <a name="11"></a>Serve the Model



** Configure and Run Mist  
You can run Mist locally or as a Docker image.


Download and compile Mist (or you can run it as a Docker image.)

```
git clone https://github.com/hydrospheredata/mist.git
cd mist
sbt -DsparkVersion=2.0.0 assembly 
```

Create Mist route.

`vi config/router.conf`

```
preventineMaintance = {
    path = "/home/walker/hydrosphere/brakePredict.py" 
    className = "Predict"
  namespace = "production" 
}


```

If you get any error about RouteConfig$RouterConfigurationMissingError then it cannot find your router.conf so put the
full path to that in default.conf:

`mist.http.router-config-path = "/home/walker/mist/mist/configs/router.conf"`

Now start Mist.

`./mist start master --config /home/walker/mist/mist/configs/default.conf --jar /home/walker/mist/mist/target/scala-2.11/mist-assembly-0.5.0.jar
`

Run the code.

`curl --header "Content-Type: application/json" -X POST http://127.0.0.1:2004/api/preventineMaintance --data '{"heatKM":[200,1000]}'`

The port number must agree with the mist.http.port = 2004 in default.conf.
 
Then it responds with the predicted value via the return ("brake is worn=", worn) statement.


`
{"success":true,"payload":{"result":["brake is worn=",1]},"errors":[],"request":{"pyPath":"/home/walker/hydrosphere/brakePredict.py","name":"brakePredict","parameters":{"heatKM":[200,20000]},"external_id":"12345678"}}`

## <a name="12"></a>Complete Code

Here is the code:


** BrakeTrain.py

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
** BrakePredict.py

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



 


