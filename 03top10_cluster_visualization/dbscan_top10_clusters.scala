val clustered = spark.read.parquet("/project_phase2/02dbscan_cluster_parquets/")
clustered.select(countDistinct(col("cluster")).as("unique_count")).collect()(0).getLong(0)
res4: Long = 156 // total 156 clusters
clustered.createOrReplaceTempView("table") 
val averageCoordinates = spark.sql("""select cluster, count(cluster) as clusterCount, avg(`ActivityLocation/LatitudeMeasure`) as averageLatitude, avg(`ActivityLocation/LongitudeMeasure`) as averageLongitude from table group by cluster order by clusterCount desc""")
averageCoordinates.show(10)

val top10 = averageCoordinates.select("cluster").as[String].collect().toSeq.take(10)
val top10clusters = clustered.filter(col("cluster").isin(top10: _*)).select(col("CharacteristicName"), col("ResultMeasureValue").cast("double"), col("ResultMeasure/MeasureUnitCode"), col("MonitoringLocationIdentifier"), col("MonitoringLocationName"), col("cluster").cast("integer"), col("ActivityLocation/LatitudeMeasure").cast("float").as("latitude"), col("ActivityLocation/LongitudeMeasure").cast("float").as("longitude"),to_timestamp(col("CleanedActivityStartDate"), "yyyy-MM-dd- HH:mm:ss").as("cleanedStartdate"), to_timestamp(col("ActivityStartDate"), "yyyy-MM-dd").as("activityStartdate"))

top10clusters.show(5)
top10clusters.printSchema()
top10clusters.coalesce(1).write.mode("overwrite").parquet("/project_phase2/03top10_cluster_visualization")

