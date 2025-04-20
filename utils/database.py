import pymongo
from pymongo import MongoClient

def get_mongodb_client(config):
    """Get MongoDB client using configuration"""
    client = MongoClient(config.MONGODB_URI)
    return client

def save_to_mongodb(collection_name, data, config):
    """Save data to MongoDB collection"""
    client = get_mongodb_client(config)
    db = client[config.MONGODB_DATABASE]
    collection = db[collection_name]

    # Insert or update the document
    if "_id" in data:
        collection.replace_one({"_id": data["_id"]}, data, upsert=True)
        return data["_id"]
    else:
        result = collection.insert_one(data)
        return result.inserted_id

def find_in_mongodb(collection_name, query, config):
    """Find documents in MongoDB collection"""
    client = get_mongodb_client(config)
    db = client[config.MONGODB_DATABASE]
    collection = db[collection_name]

    result = collection.find_one(query)
    return result

def find_many_in_mongodb(collection_name, query, config, limit=0):
    """Find multiple documents in MongoDB collection"""
    client = get_mongodb_client(config)
    db = client[config.MONGODB_DATABASE]
    collection = db[collection_name]

    cursor = collection.find(query)
    if limit > 0:
        cursor = cursor.limit(limit)

    return list(cursor)