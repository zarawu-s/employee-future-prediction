from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pyspark.ml.pipeline import Pipeline
from enum import Enum
from pyspark.sql.types import IntegerType,DoubleType
from pyspark.ml.feature import StringIndexer
from collections import Counter
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col, explode, array, lit, rand, udf
from pyspark.ml.stat import Correlation



spark = SparkSession.builder.appName("Projekat_3").getOrCreate()

data = spark.read.csv('Employee.csv', header=True, inferSchema=True)


string_indexer = StringIndexer(inputCol="Education", outputCol="IndexedEducation")
data = string_indexer.fit(data).transform(data)

string_indexer = StringIndexer(inputCol="City", outputCol="IndexedCity")
data = string_indexer.fit(data).transform(data)

string_indexer = StringIndexer(inputCol="Gender", outputCol="IndexedGender")
data = string_indexer.fit(data).transform(data)

string_indexer = StringIndexer(inputCol="EverBenched", outputCol="IndexedEverBenched")
data = string_indexer.fit(data).transform(data)

data = data.drop("Education","City","Gender","EverBenched")
#data.show(5)

data = data.select("IndexedEducation",
                    data["JoiningYear"].cast(DoubleType()).alias("JoiningYear"),
                    "IndexedCity",
                   data["PaymentTier"].cast(DoubleType()).alias("PaymentTier"),
                    data["Age"].cast(DoubleType()).alias("Age"),                 
                   "IndexedGender",
                   "IndexedEverBenched",
                   data["ExperienceInCurrentDomain"].cast(DoubleType()).alias("ExperienceInCurrentDomain"),
                   data["label"].cast(DoubleType()).alias("label"))

#data.show(5)
#print(data.columns[:-1])

assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
data = assembler.transform(data).select("features", "label")

#spark.conf.set("spark.sql.repl.eagerEval.enabled", True)
#data.show(10, truncate=False)


#resampler  = SMOTE(sampling_strategy=0.1)
#under = RandomUnderSampler(sampling_strategy=0.5)



# Split the DataFrame into minority and majority classes
minority_data = data.filter(col("label") == 1)
majority_data = data.filter(col("label") == 0)

# Calculate the size difference between the classes
minority_count = minority_data.count()
majority_count = majority_data.count()
size_diff = majority_count - minority_count
ratio = -(-majority_count//minority_count)
print("ratio: {}".format(ratio))

a = range(ratio)
# duplicate the minority rows
oversampled_df = minority_data.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows 
data = majority_data.unionAll(oversampled_df)
data.show()

# Split the DataFrame into minority and majority classes
minority_data = data.filter(col("label") == 1)
majority_data = data.filter(col("label") == 0)

# Calculate the size difference between the classes
minority_count = minority_data.count()
majority_count = majority_data.count()
size_diff = abs(majority_count - minority_count)
ratio = int(majority_count/minority_count)
print("ratio after oversampling: {}".format(ratio))

random_min_rows = data.filter(col("label") == 0).orderBy(rand()).limit(size_diff)
data = data.unionAll(random_min_rows)

minority_data = data.filter(col("label") == 1)
majority_data = data.filter(col("label") == 0)
minority_count = minority_data.count()
majority_count = majority_data.count()
size_diff = abs(majority_count - minority_count)

# # Determine the number of synthetic samples to generate
# synthetic_ratio = 1.0  # Adjust as per your requirement
# num_synthetic_samples = int(size_diff * synthetic_ratio)

# # Calculate the k-nearest neighbors
# k = 5  # Adjust as per your requirement
# x = data["features"].values
# correlation_matrix = np.corrcoef(x.reshape(-1,1), rowvar=False)
# correlation_matrix = correlation_matrix.to_list()
# knn_indices = [sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:k] for row in correlation_matrix]

# # Generate synthetic samples
# def generate_synthetic_samples(row):
#     synthetic_samples = []
#     for _ in range(num_synthetic_samples):
#         neighbor_index = knn_indices[row["label"]][int(size_diff * spark.sparkContext.nextDouble())]
#         neighbor = minority_data.collect()[neighbor_index]
#         synthetic_sample = {}
#         for feature in neighbor.asDict():
#             if feature != "label":
#                 synthetic_sample[feature] = row[feature] + spark.sparkContext.nextDouble() * (neighbor[feature] - row[feature])
#         synthetic_sample["label"] = row["label"]
#         synthetic_samples.append(synthetic_sample)
#     return synthetic_samples

# synthetic_samples = majority_data.rdd.flatMap(generate_synthetic_samples).toDF()

# # Combine the original and synthetic samples
# resampled_data = minority_data.union(synthetic_samples)

# # Show the resampled DataFrame
# resampled_data.show()


splits = data.randomSplit([0.7, 0.3], 3458)
train = splits[0]
test = splits[1]

nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
modelNb = nb.fit(train)

classifier = RandomForestClassifier(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[classifier])
modelRfc = pipeline.fit(data)

#predictions = modelNb.transform(test)
predictions = modelRfc.transform(test) #Test set accuracy = 0.8255179934569248

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

accuracy = evaluator.evaluate(predictions)
print("Test set accuracy = " + str(accuracy))

predictionAndLabels = predictions.select("prediction", "label").rdd.map(tuple)
metrics = MulticlassMetrics(predictionAndLabels)
confusionMatrix = metrics.confusionMatrix()
print("Confusion Matrix:")
print(confusionMatrix)

spark.stop()

#test commit
#test commit 2
#test commit 3
#test commit 4
#test commit 5