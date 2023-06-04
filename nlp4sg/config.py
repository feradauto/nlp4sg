import os
class Constants(object):
    def __init__(self):
        ## Directories relative to the path were you execute the files
        self.DATA_DIR = "./data"
        self.OUTPUTS_DIR = "./outputs"
        self.OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

CONSTANTS = Constants()
