Spark ML and Hydrosphere Mist Example: Preventive Maintenance

Document reference: 
Hydrosphere Mist install instructions are here
Hydrosphere Mist readme.
Other example Python code is here.

Table of Contents


Spark ML and Hydrosphere Mist Example: Preventive Maintenance	1
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

There are lots of factors that impact brake wear.  For example, brakes will wear out faster for vehicles that drive down steep hills.  But we make a linear simplified model that takes as its input:

wear_rate = x (heat) + y (kilometers driven) + intercept

We can fit this line using the least squares method and linear regression.  Then we use that equation in a binary logistic regression model that returns true (1) if the brakes are considered worn and false (0) if not.

We use this linear formula, where we have calculated the coefficients and interests by applying linear regression to sample data:

z = wear_rate = (.002 * heat) + (0.0015 * km) - 3

And plug that value into the logistic probability function:

pr =1 / (1 + e-z)

The binary logistic model, logit, requires a binary output. So if pr > 50% then logit = 1. Otherwise logit = 0.

 
Model Serving
We expose the data model as a web service for enterprise applications.  

To actually deploy this for a trucking company, a truck fleet manager would use IoT to upload truck data to a web service and call the model to get a prediction. The enterprise application would create a maintenance work order in the preventive maintenance (PM) system when a brake pad needs replacing and notifies the driver.  Then the driver returns to the truck headquarters for repairs or goes to a repair shop.

Obviously we did not write the IoT or interface to the PM system here, since that would be different for different companies and require an IoT cloud, truck, and ERP system.

For this tutorial, we write three Python programs:

1) brakeTrain.py to train the model.
2) brakePredict.py to use that model and return a prediction as to whether the brake is probably worn.
3) brakeEval.py to evaluate the model accuracy.

We start Mist first:

mist.sh master --config /home/walker/mist/mist/configs/mist.conf --jar /home/walker/mist/mist/target/scala-2.11/mist-assembly-0.4.0.jar



Data Preparation: brakeTrain.py
Since each Mist program is a web service we need to call it with HTTP.  Here is how to call the training program using CURL:

curl --header "Content-Type: application/json" -X POST http://127.0.0.1:2003/jobs --data '{"pyPath":"/home/walker/hydrosphere/brakeTrain.py", "parameters":{}, "external_id":"12345678","name":"brakeTrain"}'


pyPath is the path to the Python program you want to execute.

The three other parameters are:
 
An array of values to pass to the program.  Here we send the brake temperature in Celsius and the kilometers driven since the last brake pad replacement.

external_id is the ID of requesting client. It could set in order to dispatch the job with an asynchronous response. It is not required with a synchronous HTTP request. 

name is the name of the SparkContext to use. This parameter will be removed in the next version of the API since it is ambiguous for developer.

Here we explain parts of the code related to the Spark ML logistic regression algorithm and Hydrosphere Mist.  Below is the code so that you can copy and paste this example.  Above that we provide our explanation.

You import Hydrosphere Mist and then create a class then tell it whatever method you want to run in the constructor.  Using Mist lets us establish a SparkContext without having to start use spark-submit.

 In the constructor you tell the class what method to run.  After that we can then instantiate the class and run the model using train = Train(Mist.Job()).

import Mist

class Train:

   def __init__(self, job):
            job.sendResult(self.runModel(job))


The LogisticRegressionWithLBFGS.train Spark ML training model requires a LabeledPoint object.  In a real world sample, with actual data coming from our fleet of vehicles, or from the truck manufacturer, we would build that array using data streaming from the truck.  But here we generate random numbers for kilometers driven and brake pad temperature.  

We say that the brakes are worn based upon the probability arising from our linear regression model.  The rules of logistic regression tell us to assign the worn variable 1 if the probability is > 50% and 0 otherwise.  So the LabeledPoint object has 1 and 0 values as the first argument (i.e., the labels)  in its constructor and then an array of the heat and kilometers values in the second (i.e., the points).  Then we feed that into the train function and finally save the model on disk for use later.

a.append(LabeledPoint(worn,array([heat, km])))
lrm = LogisticRegressionWithLBFGS.train(job.sc.parallelize(a))
lrm.save(job.sc, "/tmp/brakeModel")

Here is the brakeTrain.py code:


import Mist
import os
import numpy as np
import pandas as pd
import math
from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

class Train:

    def __init__(self, job):
        job.sendResult(self.runModel(job))

    def runModel(self, job):
        val = job.parameters.values()
        heatKm = pd.DataFrame({'worn' : 0, 'heat' : np.random.random_integers(200,size=(100)), 'km' : np.random.random_integers(20000,size=(100))})
        a = []
        for index,row in heatKm.iterrows():
            heat = row['heat']
            km = row['km']
            z = (.002 * heat) + (0.0015 * km) - 3
            pr = 1 / (1 + (math.e**-z))
            worn = pr > 0.5
            a.append(LabeledPoint(worn,array([heat, km])))
            print (heat,km,z,pr,worn)
        lrm = LogisticRegressionWithLBFGS.train(job.sc.parallelize(a))


        lrm.save(job.sc, "/tmp/brakeModel")

        return 1


train = Train(Hydrosphere Mist.Job())


It echoes the response:

{"success":true,"payload":{"result":1},"errors":[],"request":{"pyPath":"/home/walker/hydrosphere/brakeTrain.py","name":"brakeTrain","parameters":{"heatKM":[200,20000]},"external_id":"12345678"}}


You can look at stdout from Hydrosphere Mist to debug any errors.
Data Ingestion: brakePredict.py
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


 
Offline Evaluation: brakeEval.py
In this code we evaluate the model by loading the saved train model and then using that to predict values based on test (real) data.  We then sum the number of correct observations divided by the sum of total observations to yield the model accuracy.

We use random readings of heat and distance driven then we use our linear function:

 z = wear_rate = (.002 * heat) + (0.0015 * km) - 3

to calculate whether the brakes are worn are not.  Then we compare that result to what the training model predicts.  We keep track of how many times the model and the actual results agreed and we divide that sum by the total number of observations to yield the accuracy of the model.

We run this code by sending this curl:

curl --header "Content-Type: application/json" -X POST http://127.0.0.1:2003/jobs --data '{"pyPath":"/home/walker/hydrosphere/brakeEval.py", "parameters":{}, "external_id":"12345678","name":"brakeEval"}'

It responds with the results, which shows an accuracy of 96%.

{"success":true,"payload":{"result":["accuracy of model=",0.96]},"errors":[],"request":{"pyPath":"/home/walker/hydrosphere/brakeEval.py","name":"brakeEval","parameters":{},"external_id":"123


Here is the brakeEval.py code:

import mist
import os
import numpy as np
import pandas as pd
import math
from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel


class Evaluate:

    def __init__(self, job):
        job.sendResult(self.runModel(job))

    def runModel(self, job):
        val = job.parameters.values()
        lrm = LogisticRegressionModel.load(job.sc, "/tmp/brakeModel")
        train = pd.DataFrame({'worn' : 0, 'heat' : np.random.random_integers(200,size=(100)), 'km' : np.random.random_integers(20000,size=(100))})

        
        a = []
        for index,row in train.iterrows():
            heat = row['heat']
            km = row['km']
            z = (.002 * heat) + (0.0015 * km) - 3
            pr = 1 / (1 + (math.e**-z))
            worn = pr > 0.5
            a.append([heat, km, worn])
         
            
        wrongObs = 0
        for heat, km, worn in a:
            predict = lrm.predict([heat, km])
            wrong = math.fabs(predict - worn) 
            wrongObs += wrong


        accuracy = (len(a) - wrongObs) / len(a)
             
        return ("accuracy of model=", accuracy)

        
      
eval = Evaluate(mist.Job())



 

 




