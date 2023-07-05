from pyspark.sql import SparkSession, Row
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, rand
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import NaiveBayes, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.pipeline import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn import neighbors
import random
import numpy as np
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.ml.linalg import Vectors
from sklearn.neighbors import KNeighborsClassifier
import datetime
from pyspark.sql.functions import when

spark = SparkSession.builder.appName("Projekat_3").getOrCreate()
sc = spark.sparkContext
data = spark.read.csv('voice.csv', header=True, inferSchema=True)

# Convert "male" to 0 and "female" to 1 in the "label" column
data = data.withColumn("label", when(data.label == "male", 0).otherwise(1))


data = data.select(data["label"].cast(DoubleType()).alias("label"))

assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")

# assembler = VectorAssembler(inputCols=data.columns[1:], outputCol="features")


# data = assembler.transform(data).select("features", "label")

data = assembler.transform(data)


minority_count = data.filter(col("label") == 1).count()
majority_count = data.filter(col("label") == 0).count()
size_diff = abs(majority_count - minority_count)
print("Size difference before balancing: {}".format(size_diff))

if(size_diff > 0) :
    # SMOTE OVERSAMPLING
    def SmoteSampling(vectorized, k = 5, minorityClass = 1, majorityClass = 0, percentageOver = 200, percentageUnder = 100):
        if(percentageUnder > 100|percentageUnder < 10):
            raise ValueError("Percentage Under must be in range 10 - 100");
        if(percentageOver < 100):
            raise ValueError("Percentage Over must be in at least 100");
        dataInput_min = vectorized[vectorized['label'] == minorityClass]
        dataInput_maj = vectorized[vectorized['label'] == majorityClass]
        feature = dataInput_min.select('features')
        feature = feature.rdd
        feature = feature.map(lambda x: x[0])
        feature = feature.collect()
        feature = np.asarray(feature)
        nbrs = neighbors.NearestNeighbors(n_neighbors=k, algorithm='auto').fit(feature)
        neighbours =  nbrs.kneighbors(feature)
        gap = neighbours[0]
        neighbours = neighbours[1]
        min_rdd = dataInput_min.drop('label').rdd
        pos_rddArray = min_rdd.map(lambda x : list(x))
        pos_ListArray = pos_rddArray.collect()
        min_Array = list(pos_ListArray)
        newRows = []
        nt = len(min_Array)
        nexs = percentageOver//100
        for i in range(nt):
            for j in range(nexs):
                neigh = random.randint(1,k)
                difs = min_Array[neigh][0] - min_Array[i][0]
                newRec = (min_Array[i][0]+random.random()*difs)
                newRows.insert(0,(newRec))
        newData_rdd = sc.parallelize(newRows)
        newData_rdd_new = newData_rdd.map(lambda x: Row(features = x, label = 1))
        new_data = newData_rdd_new.toDF()
        new_data_minor = dataInput_min.unionAll(new_data)
        new_data_major = dataInput_maj.sample(False, (float(percentageUnder)/float(100)))
        return new_data_major.unionAll(new_data_minor)

    data = SmoteSampling(data, k = 2, minorityClass = 1, majorityClass = 0, percentageOver=100)

# After oversampling minority_data, we oversample majority_data
    minority_count = data.filter(col("label") == 1).count()
    majority_count = data.filter(col("label") == 0).count()
    size_diff = abs(majority_count - minority_count)
    random_min_rows = data.filter(col("label") == 0).orderBy(rand()).limit(size_diff)
    data = data.unionAll(random_min_rows)

    minority_count = data.filter(col("label") == 1).count()
    majority_count = data.filter(col("label") == 0).count()
    size_diff = abs(majority_count - minority_count)
    print("Size difference after balancing: {}".format(size_diff))

# # splits = data.randomSplit([0.7, 0.3], 5678)
# # train = splits[0]
# # test = splits[1]

# # nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# # classifier = DecisionTreeClassifier(featuresCol="features", labelCol="label")
# # pipeline = Pipeline(stages=[classifier])

# # classifier = RandomForestClassifier(featuresCol="features", labelCol="label")
# # pipeline = Pipeline(stages=[classifier])

# # lr = LogisticRegression()
# # grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
# # evaluator = BinaryClassificationEvaluator()
# # cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator,
# #     parallelism=2)
# # modelRfc = cv.fit(data)


# # modelNb = cv.fit(data)
# # modelRfc = cv.fit(data)
# # modelJ48 = cv.fit(data)

# # modelNb = nb.fit(train)
# # modelRfc = pipeline.fit(train)
# # modelJ48 = pipeline.fit(train)

# # predictions = modelNb.transform(test) #Test set accuracy = 0.6553980370774264 // 0.6728922091782283
# # predictions = modelRfc.transform(data) #Test set accuracy = 0.8353326063249727 // 0.8585912486659552
# # predictions = modelJ48.transform(test) #Test set accuracy = 0.8173391494002181 // 0.8628601921024547

#KNN
# rdd = data.rdd.map(lambda row: (Vectors.dense(row.features.toArray()), row.label))
# X = np.array(rdd.map(lambda x: x[0]).collect())
# y = np.array(rdd.map(lambda x: x[1]).collect())

data_array = data.rdd.map(lambda row: (row.features.toArray(), row.label)).collect()
X = np.array([x[0] for x in data_array])
y = np.array([x[1] for x in data_array])





# Print the shape of X and y to check if they have the expected dimensions
print(X.shape)
print(y.shape)

# Verify the content of X and y
print(X)
print(y)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
predictions = knn.predict(X)
prediction_rdd = spark.sparkContext.parallelize(zip(predictions.tolist(), y.tolist()))
prediction_df = prediction_rdd.toDF(["prediction", "label"])

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(prediction_df)
print("Test set accuracy = " + str(accuracy)) # Test set accuracy = 0.88921875

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", metricName="areaUnderROC")
auc_roc = evaluator.evaluate(prediction_df)
print("Area under ROC curve:", auc_roc) # Area under ROC curve: 0.88921875

predictionAndLabels = prediction_df.select("prediction", "label").rdd.map(tuple)
metrics = MulticlassMetrics(predictionAndLabels)
confusionMatrix = metrics.confusionMatrix()
print("Confusion Matrix:")
print(confusionMatrix) #DenseMatrix([[3012.,  188.],
                                    #[ 521., 2679.]])


timestamp = datetime.datetime.now().strftime("%d-%m-%y %H-%M")
test_results = "Test set accuracy: {}\nArea under ROC curve: {}\nConfusion Matrix:\n{}".format(
    accuracy, auc_roc, confusionMatrix)
print(test_results)

file_name = timestamp + " test_results.txt"
with open(file_name, 'w') as file:
    file.write(test_results)



spark.stop()
