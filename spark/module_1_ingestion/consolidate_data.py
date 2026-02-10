import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Get current working directory (should be CryptX root)
current_dir = os.getcwd()
print(f"Running from: {current_dir}")

# Paths
data_path = os.path.join(current_dir, "data", "processed", "BTCUSDT", "1m")
temp_path = os.path.join(current_dir, "data", "processed", "BTCUSDT", "1m_temp")

print(f"Reading from: {data_path}")
print(f"Temp directory: {temp_path}")

# Check if path exists
if not os.path.exists(data_path):
    print(f"ERROR: Path does not exist: {data_path}")
    exit(1)

print("Path exists! ✓")
print("\nStarting consolidation...")

# Create Spark session
spark = SparkSession.builder \
    .appName("ConsolidateData") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

try:
    # Step 1: Read all existing data
    print("[1/7] Reading all parquet files...")
    df = spark.read.parquet(data_path)
    
    initial_count = df.count()
    print(f"      Found {initial_count:,} records")
    
    # Step 2: Remove duplicates
    print("[2/7] Removing duplicates...")
    df = df.dropDuplicates(["timestamp"])
    
    after_dedup = df.count()
    removed = initial_count - after_dedup
    print(f"      Removed {removed:,} duplicates")
    print(f"      Remaining: {after_dedup:,} records")
    
    # Step 3: Sort by timestamp
    print("[3/7] Sorting by timestamp...")
    df = df.orderBy("timestamp")
    
    # Step 4: Cache the dataframe to avoid re-reading files
    print("[4/7] Caching data in memory...")
    df = df.cache()
    df.count()  # Force cache
    
    # Step 5: Recalculate log returns
    print("[5/7] Recalculating log returns...")
    window = Window.orderBy("timestamp")
    df = df.withColumn("prev_close", F.lag("price_close", 1).over(window))
    df = df.withColumn("log_return", 
                      F.log(F.col("price_close") / F.col("prev_close")))
    df = df.drop("prev_close")
    
    # Step 6: Write to TEMPORARY location first (avoids race condition)
    print(f"[6/7] Writing to temporary location...")
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    
    df.coalesce(12).write \
        .mode("overwrite") \
        .partitionBy("year", "month") \
        .parquet(temp_path)
    
    print("[6/7] ✓ Temporary write complete")
    
    # Step 7: Replace old data with new data
    print(f"[7/7] Replacing old data with consolidated data...")
    
    # Remove old data
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    
    # Move temp data to final location
    shutil.move(temp_path, data_path)
    
    print("\n" + "="*60)
    print("✅ CONSOLIDATION COMPLETE!")
    print("="*60)
    print(f"Total records: {after_dedup:,}")
    print(f"Duplicates removed: {removed:,}")
    print(f"Output location: {data_path}")
    
    # Read back and show summary
    print("\n[VERIFICATION] Reading consolidated data...")
    df_final = spark.read.parquet(data_path)
    
    print("\nData by month:")
    df_final.groupBy("year", "month").count().orderBy("year", "month").show()
    
    print("\nSample records:")
    df_final.select("timestamp", "price_close", "volume_base", "log_return").show(10)
    
except Exception as e:
    print(f"\nERROR: {e}")
    # Clean up temp directory if it exists
    if os.path.exists(temp_path):
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_path)
    raise

finally:
    # Clean up
    spark.stop()
    print("\nDone!")