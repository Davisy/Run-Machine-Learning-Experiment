import os
import logging
"""
LOGGING LEVELS

DEBUG:Detailed information,, typically of interest only when diagnosis problems(this is the lowest level)

INFO:confirmation that things are working as expected 

WARNING: An indication that something unexpected happened, or indicative of some 
problem in the near future.The software is still working as expected

ERROR:Due to a more serious problem, the software has not been able to perform some function

CRITICAL: A serious error, indicating that the program itself may be unable to continue running(this is 
          the hightest level)
"""

LOGS = "../logs/"
def log(file):
    if file not in os.listdir(LOGS):
            open(os.path.join(LOGS, file), "w+").close()
    else:
        pass

    # configure logger
    logger = logging.getLogger(__name__)

    # create a file handler for output file
    file_handler = logging.FileHandler(os.path.join(LOGS, file))

    # set the logging level for log file
    logger.setLevel(logging.DEBUG)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

     #set formater to the file_handler   
    file_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)

    return logger
