from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def run_algorithm(df, training_split, testing_split):
    (df_training, df_testing) = df.randomSplit([training_split, testing_split])

    decision_tree = DecisionTreeClassifier(labelCol="label", featuresCol="features")
    model = decision_tree.fit(df_training)
    prediction = model.transform(df_testing)

    # Setas venenosas
    venenosas = prediction \
        .filter(prediction["prediction"] == 1)
    print("Setas venenosas:", venenosas.count())

    # Setas comestibles
    comestibles = prediction \
        .filter(prediction["prediction"] == 0)
    print("Setas comestibles:", comestibles.count())

    # Medimos la precisión del modelo
    evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(prediction)
    
    print("Error = %g " % (1.0 - accuracy))
    print("Acierto = %g " % accuracy)

    print("***************************************************************************************")

    # Cross validation
    grid = (ParamGridBuilder()
             .addGrid(decision_tree.maxDepth, [2, 5, 10])
             .addGrid(decision_tree.maxBins, [10, 20])
             .build())
             
    binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    cross_validator = CrossValidator(estimator=decision_tree, estimatorParamMaps=grid, evaluator=binary_evaluator,
    parallelism=2)

    dtcvModel = cross_validator.fit(df_training)
    print(dtcvModel.avgMetrics)
    dtpredictions = dtcvModel.transform(df_testing)
    print('Accuracy:', binary_evaluator.evaluate(dtpredictions))


if __name__ == "__main__":
    spark_session = SparkSession \
        .builder \
            .getOrCreate()

    # Carga de datos de setas (en CSV)
    data_frame_setas = spark_session \
        .read \
        .format("csv") \
        .option("header", True) \
        .load("data/agaricus-lepiota.data")

    # StringIndexer que convierte las features categoricas en numericas
    indexers = [StringIndexer(inputCol=column, outputCol="c"+column).fit(data_frame_setas) \
        for column in list(data_frame_setas.columns) ]
    pipeline = Pipeline(stages=indexers)
    df = pipeline \
        .fit(data_frame_setas) \
        .transform(data_frame_setas)

    # Al aplicar el indexador previamente, se borran las no numericas
    raw_features = ["l","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22"]
    for raw_feature in raw_features:
        df = df.drop(df[raw_feature])

    # Se convierten las filas del CSV en labeled points para entrenar el modelo
    df_svm = df.rdd.map(lambda x:(x[0], Vectors.dense(x[1:-1]) )).toDF(["label", "features"])

    # Se prueban varias % de training y testing
    for t in range(5):
        run_algorithm(df_svm, (70+t*5)/100, 1-((70+t*5)/100))