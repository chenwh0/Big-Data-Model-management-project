val clustered = spark.read.parquet("/project_phase2/clustered/")
val dataSeasons = clustered.withColumn(
"monthDay", date_format(col("ActivityStartDate"), "MMdd").cast("int"))
.withColumn("seasonID",
when(col("monthDay") >= 301 && col("monthDay") <= 531, 0)
.when(col("monthDay") >= 601 && col("monthDay") <= 831, 1)
.when(col("monthDay") >= 901 && col("monthDay") <= 1130, 2)
.otherwise(3)
)
.withColumn("season",
when(col("seasonID") === 0, "spring")
.when(col("seasonID") === 1, "summer")
.when(col("seasonID") === 2, "fall")
.otherwise("winter")
)
.drop("month_day") // have conditional for dates (in format yyyy-MM-dd but ignore year)
dataSeasons.createOrReplaceTempView("table")

val nitrogen = spark.sql("""select * from table where CharacteristicName='Nitrogen'""")
val temperature = spark.sql("""select * from table where CharacteristicName='Temperature, water'""")
val pH = spark.sql("""select * from table where CharacteristicName='pH'""")
val dissolved_oxygen = spark.sql("""select * from table where CharacteristicName='Dissolved oxygen (DO)'""")
val phosphorus = spark.sql("""select * from table where CharacteristicName='Phosphorus'""")
val total_suspended_solids = spark.sql("""select * from table where CharacteristicName='Total suspended solids'""")

nitrogen.write.mode("overwrite").parquet("/project_phase2/nitrogen")
temperature.write.mode("overwrite").parquet("/project_phase2/temperature")
pH.write.mode("overwrite").parquet("/project_phase2/pH")
dissolved_oxygen.write.mode("overwrite").parquet("/project_phase2/dissolved_oxygen")
phosphorus.write.mode("overwrite").parquet("/project_phase2/phosphorus")
total_suspended_solids.write.mode("overwrite").parquet("/project_phase2/total_suspended_solids")

