import mist
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


train = Train(mist.Job())
