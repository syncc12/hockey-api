from pymongo import MongoClient

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['hockey']

# Get database statistics
db_stats = db.command("dbStats")

# The 'dataSize' field contains the size of the data in bytes
data_size = db_stats.get('dataSize', 0)

# The 'storageSize' field contains the size of the data on disk in bytes
storage_size = db_stats.get('storageSize', 0)

# The 'indexSize' field contains the total size of all indexes in bytes
index_size = db_stats.get('indexSize', 0)

# Print the sizes
print(f"Data Size: {data_size / (1024 ** 2):.2f} MB")
print(f"Storage Size: {storage_size / (1024 ** 2):.2f} MB")
print(f"Index Size: {index_size / (1024 ** 2):.2f} MB")

# Total size (data + indexes)
total_size = data_size + index_size
print(f"Total Size (Data + Indexes): {total_size / (1024 ** 2):.2f} MB")
