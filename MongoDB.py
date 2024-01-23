import pymongo
from pymongo import MongoClient
import datetime
from utils import gaia_utils

DATABASE = 'GAIA'
COLLECTION_FEEDBACK = 'Feedback'
SUB_COLLECTION_USERS = 'Users'
SUB_COLLECTION_MODELS = 'Models'
SUB_COLLECTION_STAT = 'Statistics'
COLLECTION_HISTORY = 'HISTORY'
COLLECTION_UNITS = 'Units'

def GetStoreMsgFromModel(model, UserID):
    result = {}
    res = []

    client = gaia_utils.ClientDB
    mydb = client[DATABASE]
    coll = mydb[COLLECTION_HISTORY]

    cursor = coll.find({})
    for document in cursor:
        result = document['Models'].get(model, {})

    return result

def InsertToDB(collection, data, UserID):
    client = gaia_utils.ClientDB
    mydb = client[DATABASE]
    coll = mydb[collection]
    result = True
    QntFeedbacks = 0
    PositiveFeedback = 0
    NegativeFeedback = 0
    if coll.name == COLLECTION_FEEDBACK:
        Find = False
        cursor = coll.find({})
        for document in cursor:
            for key, value in document[SUB_COLLECTION_USERS].items():
                for fed in value:
                    QntFeedbacks += 1
                    if fed["Feedback"] == "POSITIVE":
                        PositiveFeedback += 1
                    else:
                        NegativeFeedback += 1
                if key == UserID:
                    Find = True

        if Find == True:
            QntFeedbacks += 1
            if data["Feedback"] == "POSITIVE":
                PositiveFeedback += 1
            else:
                NegativeFeedback += 1
            coll.delete_one(document)
            document[SUB_COLLECTION_USERS][UserID].append(data)
            document[SUB_COLLECTION_STAT]["Quantity Users"] = str(len(document[SUB_COLLECTION_USERS]))
            document[SUB_COLLECTION_STAT]["Quantity Feedbacks"] = str(QntFeedbacks)
            document[SUB_COLLECTION_STAT]["Positive Feedback"] = str(PositiveFeedback)
            document[SUB_COLLECTION_STAT]["Negative Feedback"] = str(NegativeFeedback)
            document[SUB_COLLECTION_STAT]["Others"] = "0"
            x = coll.insert_one(document)
        elif Find == False:
            QntFeedbacks += 1
            if data["Feedback"] == "POSITIVE":
                PositiveFeedback += 1
            else:
                NegativeFeedback += 1
            coll.delete_one(document)
            lst = [data]
            document[SUB_COLLECTION_USERS][UserID] = lst
            document[SUB_COLLECTION_STAT]["Quantity Users"] = str(len(document[SUB_COLLECTION_USERS]))
            document[SUB_COLLECTION_STAT]["Quantity Feedbacks"] = str(QntFeedbacks)
            document[SUB_COLLECTION_STAT]["Positive Feedback"] = str(PositiveFeedback)
            document[SUB_COLLECTION_STAT]["Negative Feedback"] = str(NegativeFeedback)
            document[SUB_COLLECTION_STAT]["Others"] = "0"
            x = coll.insert_one(document)

    elif coll.name == COLLECTION_HISTORY:
        Find = False
        cursor = coll.find({})
        for document in cursor:
            Find = True
            coll.delete_one(document)
            document[SUB_COLLECTION_MODELS][UserID] = data
            x = coll.insert_one(document)
            # for key, value in document[SUB_COLLECTION_MODELS].items():
            #     if key == UserID:
            #         Find = True
            #         coll.delete_one(document)
            #         document[SUB_COLLECTION_MODELS][key].append(data)
            #         x = coll.insert_one(document)
        if Find == False:
            coll.delete_one(document)
            lst = [data]
            #document[SUB_COLLECTION_MODELS][UserID] = lst
            document[SUB_COLLECTION_MODELS][UserID] = data
            x = coll.insert_one(document)

    return result

def InitDB():
    client = MongoClient()
    gaia_utils.ClientDB = client
    dblist = client.list_database_names()
    if DATABASE in dblist:
        print("the database exists.")
        # dev = {
        #     "UserIP": '1.1.1.1',
        # }
        # dev = InsertToDB(COLLECTION_FEEDBACK, dev, dev["UserIP"])
    else:
        dev = {
            SUB_COLLECTION_USERS: {},
            SUB_COLLECTION_STAT: {
                "Quantity Users": "0",
                "Quantity Feedbacks": "0",
                "Positive Feedback": "0",
                "Negative Feedback": "0",
                "Others": "",
            },
        }
        mydb = client[DATABASE]
        coll_feedback = mydb[COLLECTION_FEEDBACK]
        coll_feedback.insert_one(dev)

        dev = {
            SUB_COLLECTION_MODELS: {},
            SUB_COLLECTION_STAT: {},
        }
        mydb = client[DATABASE]
        coll_history = mydb[COLLECTION_HISTORY]
        coll_history.insert_one(dev)

        # cursor = coll_feedback.find({})
        # for document in cursor:
        #     print(document)
        #     if document['UserID'] == '67891':
        #         new_val = {
        #             "Updated": datetime.datetime.utcnow(),
        #             "Param_1": '0777',
        #         }
        #         coll_feedback.update_one({"_id": document['_id']},
        #                               {"$set": new_val},
        #                               upsert=True
        #                               )

InitDB()