Spark ML and Hydrosphere Mist Example: Preventive Maintenance


 




Document reference: 
Hydrosphere Mist install instructions are here
Hydrosphere Mist readme.
Other example Python code is here.
“Brake Wear and Performance Test Final Report” from US Department of Transportation.
Read this tutorial for reference and Spark ML lib documentation since MapR tutorial uses old Spark RDD API


Table of Contents




Spark ML and Hydrosphere Mist Example: Preventive Maintenance	1
Business Assessment: Use Case Background	2
Vehicle Fleets and Analytics	2
Brake Failure Prediction	2
Brake Pad Maintenance	2
Model Serving	3
Data Preparation: brakeTrain.py	4
Data Ingestion: brakePredict.py	6
Offline Evaluation: brakeEval.py	8






Here we provide an example in Python of how to use Hydrosphere Mist with Spark ML (machine learning library).  We take this example from the field of preventive maintenance (PM) as explained below.


This is a tutorial.  Here is what you will learn:
 
Use Hydrosphere Mist to run a Spark job.
Create and train a logistic regression model using Spark ML and Python.
Write a webservice in Python using Hydrosphere Mist that predicts when brakes need to be replaced on a heavy truck.
Train model and save it.  Then use it for prediction queries.
Evaluate the model for accuracy. 


Below we discuss the code in depth.  But first we give a use case for why this is needed.


Business Assessment: Use Case Background
PM was one of the early adopters of big data analytics and machine learning and IoT (Internet of Things) because it is so simple to conceive and implement for that use case.  Calculating when a machine needs maintenance is a problem that fits neatly into a predictive algorithm. This is because machine wear is a function of time and usage.  


Vehicle Fleets and Analytics
IoT-equipped trucks send data from vehicles using a cellular or satellite signal either as a stream or in bursts.  With IoT, trucks are fit with sensors and GPS trackers that measure heat, vibration, distance travelled, speed, etc.  These are attached to the engine, brakes, transmission, refrigerated trailer, etc.


Companies gather and study this data to operate their vehicles in the safest and lowest cost manner possible.  For example, sensors on the engine can tell whether the engine has a problem.  It is the goal of PM to fix a device before it breaks as waiting until it breaks is expensive as the engine, brake assembly, or drive train can be destroyed and the vehicle taken out of service for a longer period of time than if it is properly maintained


Brake Failure Prediction
A heavy truck with 18 wheels has a unique preventive maintenance problem to solve, and that is knowing when to change brakes.  Trucks needs to know when to replace their brakes so that they do not have an accident or destroy the brake rotor, which is the metal part of the assembly.  If they wait too long the brake pad will destroy the rotor as metal rubs up against metal.   


The driver cannot be expected to check every brake every time they stop.  And if the company just changes brakes based on some preset schedule then they are wasting money, because they might be changing them too often. So it is preferred to write some mathematical or statistical model to predict when brakes should be changed.  


Brake Pad Maintenance
Brake pads are metal shavings held together by a resin. The brake applies pressure to the pad to force it down on the rotor, which is a metal disk connected to a truck’s axles.  The pad is designed to wear out over time.  It has to be softer than the rotor, so that it does not damage the rotor.   When the brake pad wears down, heat will go up because there is more friction.  And the further a vehicle has been driven the more its brakes will have worn down.


We contacted an engineer from Volvo and he verified that this model would work as a teaching exercise as it seems reasonable to correlate heat and distance driven with wear.  To get a more accurate model we would have to use something like this winning paper from IDA Industrial Challenge, which was a competition made by Scana trucking company.


There are lots of factors that impact brake wear.  For example, brakes will wear out faster for vehicles that drive down steep hills.   


We do not have any actual sample data.  So we generated some sample date using this rough model:




z = wear_rate = =(0.003 *heat)+(0.004*kilometers)-78


This shows whether the brakes are worn out given the kilometers driven and the maximum heat generated during gathering the sample.


We plug that value into the logistic probability function:


pr =1 / (1 + e-z)




The binary logistic model, logit, requires a binary output. So if pr > 50% then worn = 1. Otherwise logit = 0. If worn = 1 then time to change brake pads.




worn
km
heat
z
pr
1
20,000
240
2.72
0.938197
0
5,000
98
-57.706
0.000000
1
50,000
140
122.42
1.000000
0
8,000
260
-45.22
0.000000
0
15,966
263
-13.347
0.000002
1
27,110
201
31.043
1.000000
0
16,018
189
-13.361
0.000002
1
28,792
232
37.864
1.000000
1
22,002
201
10.611
0.999975
0
10,227
175
-36.567
0.000000
0
13,663
183
-22.799
0.000000
0
8,273
264
-44.116
0.000000


 
Model Serving
We expose the data model as a web service for enterprise applications.  


To actually deploy this for a trucking company, a truck fleet manager would use IoT to upload truck data to a web service and call the model to get a prediction. The enterprise application would create a maintenance work order in the preventive maintenance (PM) system when a brake pad needs replacing and notifies the driver.  Then the driver returns to the truck headquarters for repairs or goes to a repair shop.


Obviously we did not write the IoT or interface to the PM system here, since that would be different for different companies and require an IoT cloud, truck, and ERP system.


For this tutorial, we write three Python programs:

1) brakeTrain.py to train the model and calculate its efficiency..
2) brakePredict.py to use that model and return a prediction as to whether the brake is probably worn.
 
We can run Mist as a Docker image of install it locally.  We install it locally and run Mist like this:
 


mist.sh master --config /home/walker/mist/mist/configs/mist.conf --jar /home/walker/mist/mist/target/scala-2.11/mist-assembly-0.4.0.jar




Data Preparation: brakeTrain.py
Download the training data from Github here.


Copy the code below into PySpark and run it there.

 



The LogisticRegressionWithLBFGS.train Spark ML training model requires a LabeledPoint object.  In a real world sample, with actual data coming from our fleet of vehicles, or from the truck manufacturer, we would build that array using data streaming from the truck.  But here we generate random numbers for kilometers driven and brake pad temperature.  


We say that the brakes are worn based upon the probability arising from our linear regression model.  The rules of logistic regression tell us to assign the worn variable 1 if the probability is > 50% and 0 otherwise.  So the LabeledPoint object has 1 and 0 values as the first argument (i.e., the labels)  in its constructor and then an array of the heat and kilometers values in the second (i.e., the points).  Then we feed that into the train function and finally save the model on disk for use later.


a.append(LabeledPoint(worn,array([heat, km])))
lrm = LogisticRegressionWithLBFGS.train(job.sc.parallelize(a))
lrm.save(job.sc, "/tmp/brakeModel")


Here is the brakeTrain.py code:




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


 
Data Ingestion: brakePredict.py
This job is exposed as a web service by Mist.   




 In the constructor you tell the class what method to run.  After that we can then instantiate the class and run the model using train = Train(Mist.Job()).


import Mist


class Train:


   def __init__(self, job):
            job.sendResult(self.runModel(job))


Here is how to call the predictive model using CURL:


curl --header "Content-Type: application/json" -X POST http://127.0.0.1:2003/jobs --data '{"pyPath":"/home/walker/hydrosphere/brakePredict.py", "parameters":{"heatKM":[200,20000]}, "external_id":"12345678","name":"brakePredict"}'




Here we take the two arguments heat and kilometers driven that we passed to the program, load the training model that we saved about, and then run the predict function.  That returns a 1 or 0 indicating whether the brakes are worn or not.


heat = pylist[0]
km = pylist[1]
lrm = LogisticRegressionModel.load(job.sc, "/tmp/brakeModel")
worn = lrm.predict([heat, km])


Here is the brakePredict.py code:


import Mist
import os
import numpy as np
import pandas as pd
import math
from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel


class Predict:
 


    def __init__(self, job):
        job.sendResult(self.runModel(job))


    def runModel(self, job):
        val = job.parameters.values()
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
        lrm = LogisticRegressionModel.load(job.sc, "/tmp/brakeModel")
        worn = lrm.predict([heat, km])
        return ("brake is worn=", worn)
      


predict = Predict(Mist.Job())


Then it responds with the predicted value via the return ("brake is worn=", worn) statement.


{"success":true,"payload":{"result":["brake is worn=",1]},"errors":[],"request":{"pyPath":"/home/walker/hydrosphere/brakePredict.py","name":"brakePredict","parameters":{"heatKM":[200,20000]},"external_id":"12345678"}}




 




