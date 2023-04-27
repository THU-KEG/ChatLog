import socket
from urllib.parse import quote_plus
import requests
import configparser
import os

curPath = os.path.dirname(os.path.realpath(__file__))
cfgPath = os.path.join(curPath, "conf.ini")


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


class Config:
    cf = configparser.ConfigParser()
    cf.read(cfgPath)
    session_cooling_time = 30  # minutes

    def __init__(self):
        self.local_ip = get_host_ip()
        self.set_plm_url()
        self.set_tools()
        self.set_database()
        self.set_server()
        self.set_mongo()

    def set_database(self):
        section = "HISTORY_STORAGE"
        self.MEM_method = self.cf.get(section, "type")

    def set_server(self):
        section = "SERVER"
        self.server_ip = self.cf.get(section, "server_ip")
        # each sub tasks of detection
        # knowledge extraction
        self.ke_server_port = self.cf.get(section, "ke_server_port")
        self.ke_server_url = f"http://{self.server_ip}:{self.ke_server_port}"
        self.knowledge_extraction_route = self.cf.get(section, "knowledge_extraction")
        self.knowledge_extraction_api = self.ke_server_url + self.knowledge_extraction_route
        # classification
        self.classify_server_port = self.cf.get(section, "classify_server_port")
        self.classify_server_url = f"http://{self.server_ip}:{self.classify_server_port}"
        self.classification_route = self.cf.get(section, "classification")
        self.classification_api = self.classify_server_url + self.classification_route

    def set_tools(self):
        section = "TOOL"
        self.tool_api_ip = self.cf.get(section, "tool_api_ip")
        self.faq_port = self.cf.get(section, "faq_port")
        self.faq_route = self.cf.get(section, "faq_route")
        self.faq_col = self.cf.get(section, "faq_col")
        self.qagen_port = self.cf.get(section, "qagen_port")
        self.qagen_route = self.cf.get(section, "qagen_route")
        self.sbert_model = self.cf.get(section, "sbert_model")
        self.sentsim_port = self.cf.get(section, "sentsim_port")
        self.sentsim_route = self.cf.get(section, "sentsim_route")
        self.sentsim_api = f"http://{self.tool_api_ip}:{self.sentsim_port}{self.sentsim_route}"
        self.faq_api = f"http://{self.tool_api_ip}:{self.faq_port}{self.faq_route}"
        self.qagen_api = f"http://{self.tool_api_ip}:{self.qagen_port}{self.qagen_route}"

    def set_plm_url(self):
        section = "PLM"
        self.plm_ip = self.cf.get(section, "ip_address") or self.local_ip
        glm_port = self.cf.get(section, "glm_port") or 8888
        glm_url = "http://{ip_address}:{glm_port}/glm".format(ip_address=self.plm_ip, glm_port=glm_port)
        self.glm_query_api = self.cf.get(section, "glm_api") or glm_url
        self.glm_130b_query_api = self.cf.get(section, "glm_130b_query_api")
        self.default_plm_api = self.glm_query_api

    def set_mongo(self):
        user = self.cf.get("MONGO", "user")
        password = self.cf.get("MONGO", "password")
        host = self.cf.get("MONGO", "host")
        port = self.cf.get("MONGO", "port")
        self.dbname = self.cf.get("MONGO", "dbname")
        dbname = self.dbname
        self.mongo_chatbot_uri = f"mongodb://{user}:{password}@{host}:{port}/?authSource={dbname}"
        return self.mongo_chatbot_uri


CONFIG = Config()
