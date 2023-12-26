import pymongo
from pymongo import MongoClient
import datetime

client = MongoClient()
client.drop_database()
mydb = client["database"]
collection = mydb.create_collection["collection"]

dblist = client.list_database_names()
if "database" in dblist:
    print("the database exists.")
else:
    print("the database not exists.")