import org.apache.spark.sql.functions._

val characteristicName = "total_suspended_solids" // CHANGE THIS TO DIFFERENT CHARACTERSITICNAMES

val path = "/project_phase2/" + characteristicName
val output_path = "/project_phase2/summarized/" + characteristicName
val data = spark.read.parquet(path)
val dataWithYear = data.withColumn("year", year(col("ActivityStartDate")))
val yearlySummary = dataWithYear.groupBy("year").agg(expr("percentile_approx(cast(ResultMeasureValue as double), 0.5)").alias("medianYearValue"), min(col("ResultMeasureValue").cast("double")).as("minYearValue"), max(col("ResultMeasureValue").cast("double")).as("maxYearValue"), avg(col("ResultMeasureValue").cast("double")).as("avgYearValue"), avg(log1p(col("ResultMeasureValue").cast("double"))).as("logAvgYearValue"))

yearlySummary.orderBy("year").show(25)
val result = dataWithYear.join(yearlySummary, Seq("year"), "left")
result.show(2)
result.write.mode("overwrite").parquet(output_path)
