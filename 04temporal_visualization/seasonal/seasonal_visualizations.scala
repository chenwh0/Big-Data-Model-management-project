val characteristicName = "total_suspended_solids" // Change to pH, temperature, phosphorus, or other characteristics
val path = "/project_phase2/summarized/" + characteristicName
val output_path = "/project_phase2/summarized/seasonal/" + characteristicName + "/"
val data = spark.read.parquet(path).withColumn("value", col("ResultMeasureValue").cast("double"))
val seasonalitySummary = data.groupBy("seasonID", "year").agg(expr("percentile_approx(value, 0.5)").alias("medianSeasonValue"),min("value").alias("minSeasonValue"),max("value").alias("maxSeasonValue"),avg("value").alias("avgSeasonValue"))
val result = data.join(seasonalitySummary, Seq("seasonID", "year"), "left")
result.show(5)

result.write.mode("overwrite").parquet(output_path)
