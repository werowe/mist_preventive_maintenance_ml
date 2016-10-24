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
lrm.save(mist.job.sc, "/tmp/brakeModel")


p = sc.parallelize(a)

valuesAndPreds = p.map(lambda p: (p.label, lrm.predict(p.features)))


accurate = 1 - valuesAndPreds.map(lambda (v, p): math.fabs(v-p)).reduce(lambda x, y: x + y) / valuesAndPreds.count()

 


