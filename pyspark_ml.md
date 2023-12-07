## Machine learning in pyspark

### Regression

#### Imports
````py
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

#### Vec
assembler = VectorAssembler(inputCols=<<list of columns>>, outputCol='features')
output = assembler.transform(<<dataframe>>)

final_data = output.select('features', <<label_column>>)

train_data, test_data = final_data.randomSplit([0.7,0.3])

lr = LinearRegression(labelCol=<<label_column>>)
lr_model = lr.fit(train_data)

test_results = lr_model.evaluate(test_data)

test_results.rootMeanSquaredError
test_results.r2

predictions = lr_model.transform(unlabeled_data)
