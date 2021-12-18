import logging
import os


class Logger(object):
    """
    set logger

    """

    def __init__(self, logger_path):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logfile = logging.FileHandler(logger_path)
        #
        self.logfile.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s:%(levelname)s - %(message)s')
        self.logfile.setFormatter(formatter)
        self.logdisplay = logging.StreamHandler()
        #
        self.logdisplay.setLevel(logging.INFO)
        self.logdisplay.setFormatter(formatter)
        self.logger.addHandler(self.logfile)
        self.logger.addHandler(self.logdisplay)

    def get_logger(self):
        return self.logger
