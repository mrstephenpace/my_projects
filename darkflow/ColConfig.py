__author__      = "Rahul Vishwakarma"


import os
import sys
import json
import datetime

"""
    Data structure 
"""
class ColDict(dict):
    def __getattr__(self,attr):
        return self[attr]
    def __setattr__(self,attr,value):
        self[attr]=value


"""
    Configuration 
"""
class ColConfig():
    """ Configuration for Image Collection Module """
    # Target address
    PUB = "pub"
    PUB_IP = "ip"
    PUB_PORT = "port"

    IMG = "img"
    IMG_SOURCE = "source" # "cam" "file"
    IMG_PATH = "path"
    IMG_HEIGHT = "height"
    IMG_WIDTH = "width"
    IMG_VIDEO_SAVE_PATH = "videoSavePath"
    IMG_IMAGE_SAVE_PATH = "imageSavePath"
    IMG_VIDEO_PUB_INTERVAL = "videoInterval" # Interval in seconds
    IMG_IMAGE_PUB_INTERVAL = "imageInterval" # Interval in seconds
    IMG_FPS = "fps"

    VEH = "vehicle"
    VEH_URL = "url"

    SUB = "sub"
    SUB_PORT = "port"

    DATABASE = "database"
    DATABASE_URL = "url"

    config = {'key': 'value'}
    
    def __init__(self):
        self.loadConfig()

    """ Write configuration to file  """
    def writeConfig(self):
        with open('config.json', 'w') as f:
            json.dump(self.config, f)

    """ Read configuration to file  """
    def loadConfig(self):
        with open('config.json', 'r') as f:
            self.config = json.load(f)
            print("\nConfig loaded")
    
    def getConfig(self, key):
        return self.config[key]
    
    def setConfig(self, key, value):
        self.config[key] = value
    
    def getConfigs(self):
        return self.config



def main():
    conf = ColConfig()
    vals = conf.getConfig(ColConfig.DATABASE)
    print(vals)

if __name__ == '__main__':
    main()