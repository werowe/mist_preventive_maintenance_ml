import pandas as pd
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
import locale
locale.setlocale( locale.LC_ALL, 'en_US.UTF-8' ) 

df = pd.read_csv('/root/hydrosphere/brakedata.csv', header=None, sep= ' ')
slice=df.ix[:,1:3]

slice.columns=['mileage','weight','efficiency']

a = [] 

def parsePoint(m,w,e):
    worn = e < 0.5
    return LabeledPoint(worn, [mileage, weight])

for row in slice.itertuples():
	mileage = locale.atof(getattr(row, 'mileage'))
	weight = locale.atof(getattr(row, 'weight'))
	efficiency = getattr(row,'efficiency')
	lp = parsePoint (mileage, weight, efficiency)
	a.append(lp)

lrm = LogisticRegressionWithLBFGS.train(sc.parallelize(a))

for x in a:
    print (x.label, x.features)

p = sc.parallelize(a)

valuesAndPreds = p.map(lambda p: (p.label, lrm.predict(p.features)))

accurate = valuesAndPreds.map(lambda (v, p): math.fabs(v-p)).reduce(lambda x, y: x + y) / valuesAndPreds.count()

 
