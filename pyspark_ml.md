## Machine learning in pyspark

### Data preparation

#### Imports
````py
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
````
#### Indexation of  categorical data
````py
indexer = StringIndexer(inputCol=<<column>>, outputCol=<<indexed_column>>)
output = indexer.fit(<<dataframe>>).transform(<<dataframe>>)
````
#### OneHot Encoding
````py
encoder = OneHotEncoder(inputCol=<<indexed_column>>, outputCol='vec_column')
output = encoder.fit(<<dataframe>>).transform(<<dataframe>>)
````
#### Vectorize the features
````py
assembler = VectorAssembler(inputCols=<<list of columns>>, outputCol='features')
output = assembler.transform(<<dataframe>>)
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

### Logistic Regression

#### Imports
````py
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
````
#### Initialize the model and fit the data
````py
log_reg = LogisticRegression(featuresCol='features', labelCol=<<label_column>>)
````
#### Fit the model
````py
log_reg_model = log_reg(train_data)
results = log_reg_model.transform(test_data)
````
#### Evaluate the model
````py
my_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol=<<label_column>>)
AUC = my_eval.evaluate(<<results>>)
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
# AUC by default
binary_eval = BinaryClassificationEvaluator(labelCol=<<label_col>>)
# for accuracy
multi_eval = MulticlassClassificationEvaluator(labelCol=<<label_col>>, metricName='accuracy')

AUC = binary_eval.evaluate(<<predictions>>)
acc = multi_eval.evaluate(<<predictions>>)
````
#### Obtain predictions
````py
predictions = model.transform(unlabeled_data)
````

### K-Means Clustering

#### Imports
````py
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
````
#### Initialize the model
````py
kmeans = KMeans(featuresCol='scaledFeatures', k=<<n>>, seed=<<m>>)
````
#### Fit the model
````py
model = kmeans.fit(<<final_data>>)
````
#### Evaluate the model
````py
# Make predictions 
predictions = model.transform(<<final_data>>)
# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette score: ", silhouette)
````
### Recommender Systems

#### Imports
````py
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
````
#### Initialize the model
````py
als = ALS(maxIter=<<5>>, regParam=<<0.01>>, userCol=<<userId>>,itemCol=<<itemId>>,ratingCol=<<rating>>)
````
#### Fit the model
````py
model = als.fit(train_data)
````
#### Evaluate the model
````py
# Make predictions 
predictions = model.transform(test_data)
# Evaluate
evaluator = RegressionEvaluator(metricName='rmse', labelCol=<<rating>>, predictionCol='prediction')
rmse = evaluator.evaluate(predictions)
print("rmse: ", rmse)
````


