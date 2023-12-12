## Machine learning in pyspark

### Data preparation

#### Imports
````py
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
````
#### Vectorize the features
````py
assembler = VectorAssembler(inputCols=<<list of columns>>, outputCol='features')
output = assembler.transform(<<dataframe>>)
````
#### Indexation of  categorical data
````py
indexer = StringIndexer(inputCol=<<column>>, outputCol=<<indexed_column>>)
output = indexer.fit(<<dataframe>>).transform(<<dataframe>>)
````
#### Create a dataframe of features and labels
````py
final_data = output.select('features', <<label_column>>)
````
#### Split data
````py
train_data, test_data = final_data.randomSplit([0.7,0.3])
````

### Regression

#### Imports
````py
from pyspark.ml.regression import LinearRegression
````
#### Initialize the model and fit the data
````py
lr = LinearRegression(labelCol=<<label_column>>)
lr_model = lr.fit(train_data)
````
#### Evaluate the model
````py
test_results = lr_model.evaluate(test_data)
test_results.rootMeanSquaredError
test_results.r2
````
#### Obtain predictions
````py
predictions = lr_model.transform(<<unlabeled_data>>)
````

### Decision Trees and Random Forest

#### Imports
````py
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
````
#### Initialize the model
````py
dtc = DecisionTreeClassifier(labelCol=<<label_column>>, featuresCol='features')
rfc = RandomForestClassifier(labelCol=<<label_column>>, featuresCol='features')
gbt = GBTClassifier(labelCol=<<label_column>>, featuresCol='features')
````
#### Fit the model
````py
dtc_model = dtc.fit(train_data)
rfc_model = rfc.fit(train_data)
gbt_model = gbt.fit(train_data)

dtc_preds = dtc_model.transform(test_data)
rfc_preds = rfc_model.transform(test_data)
gbt_preds = gbt_model.transform(test_data)
````
#### Evaluate the model
````py
# rfc by default
binary_eval = BinaryClassificationEvaluator(labelCol=<<label_col>>)
# for accuracy
multi_eval = MulticlassClassificationEvaluator(labelCol=<<label_col>>, metricName='accuracy')

rfc = binary_eval.evaluate(<<predictions>>)
acc = multi_eval.evaluate(<<predictions>>)
````
#### Obtain predictions
````py
predictions = model.transform(unlabeled_data)


