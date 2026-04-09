val data = spark.read.option("header","true").csv("/project_phase2/waterqualityMO2000to2025.csv") 
data.createOrReplaceTempView("table") 
val quality_indicators = spark.sql("""select CharacteristicName, ResultMeasureValue, `ResultMeasure/MeasureUnitCode`, MonitoringLocationIdentifier, MonitoringLocationName, `ActivityLocation/LatitudeMeasure`, `ActivityLocation/LongitudeMeasure`, CleanedActivityStartDate, ActivityStartDate from table where CharacteristicName in ('Nitrogen', 'Temperature, water', 'pH', 'Dissolved oxygen (DO)', 'Phosphorus', 'Total suspended solids')""")
quality_indicators.coalesce(1).write.mode("overwrite").parquet("/project_phase2/quality_indicators")

quality_indicators.show(5) 
quality_indicators.count() # count all rows
quality_indicators.groupBy("CharacteristicName").count().show() # count # of rows per CharacteristicName 

