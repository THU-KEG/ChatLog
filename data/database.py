#!/usr/bin/env python
# coding=utf-8
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd

TEST_VERSION = "ChatGPT"


class MongoDB(object):

    def __init__(self, db_name='edubot', collection_name='glm_base', url=None):
        if url:
            self.client = MongoClient(url)
        else:
            self.client = MongoClient(f"mongodb://tsq:liverpool@localhost:27017/?authSource={db_name}")
        # 指定数据库
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    # 添加一条数据
    def add_one(self, data):
        result = self.collection.insert_one(data)
        print(result)

    # 添加多条
    def add_many(self, data):
        result = self.collection.insert_many(data)
        print(result)

    # 获取一条数据
    def get_one(self):
        return self.collection.find_one()

    # 获取多条数据
    def get_many(self):
        return self.collection.find()

    # 通过条件获取
    def get_data(self, data, projection_fields=None):
        if projection_fields:
            return self.collection.find(data, projection=projection_fields)
        return self.collection.find(data)

    # 单条更新
    def up_one(self, query, data):
        result = self.collection.update_one(query, data)
        print(result)

    # 多条更新
    def up_many(self, query, data):
        result = self.collection.update_many(query, data, True)
        print(result)

    # 删除数据
    def del_one(self, query):
        result = self.collection.delete_one(query)
        print(result)

    def del_many(self, query):
        result = self.collection.delete_many(query)
        print(result)

    def get_size(self):
        return self.collection.find().count()


def dump(version):
    mdb = MongoDB(collection_name=version)
    res = mdb.get_many()
    res_lst = list(res)
    df = pd.DataFrame(res_lst)
    df.to_csv(f'/data/tsq/CK/general_dialogue_test/{version}_test_{len(res_lst)}.csv')


if __name__ == '__main__':
    # mdb = MongoDB()
    # 添加
    # mdb.add_one({"title": "java", "content": "教育"})
    # dt = [
    #     {"title": "c++", "content": "C++"},
    #     {"title": "php", "content": "PHP"},
    # ]
    # mdb.add_many(dt)

    # 获取
    # result = mdb.get_one()
    # print(result)
    # 获取多条
    # res = mdb.get_many()
    # for da in res:
    #     print(da)

    # 条件获取
    # query = {"title": "python"}
    # res = mdb.get_data(query)
    # for da in res:
    #     print(da)

    # 通过_id查询导入包from bson.objectid import ObjectId
    # query = {"_id": ObjectId("625410ab39d8d7eec64e2c90")}
    # res = mdb.get_data(query)
    # for da in res:
    #     print(da)

    # 更新一条
    # q = {"title": "php"}
    # d = {"$set": {"title": "c#", "content": "C#"}}
    # mdb.up_one(q, d)

    # 更新多条
    # q = {"title": "php"}
    # d = {"$set": {"title": "python", "content": "MongoDB"}}
    # mdb.up_many(q, d)

    # 删除单条
    # q = {"title": "c#"}
    # mdb.del_one(q)

    # 删除多条
    # q = {"title": {"$regex": "c++"}}
    # mdb.del_many(q)
    dump(TEST_VERSION)
