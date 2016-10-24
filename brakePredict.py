import mist
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
        worn = lrm.predict([km,heat])
        return ("brake is worn=", worn)
      

predict = Predict(mist.Job())

 
