import os
import sys
import glob
import time
import types
import logging
import pandas as pd
from pathlib import Path
from logging.config import dictConfig


def log_blank(self):
    """ Function to add to a logger object
    Log a blank line
    """
    # Create a temporary formatter which doesn't have any format
    blank_formatter = logging.Formatter('')

    # Retrieve the actual formatters and replace them by the blank one
    old_formatters = []
    for handler in self.handlers:
        old_formatters.append(handler.formatter)
        handler.setFormatter(blank_formatter)

    # Write an empty message: with the blank formatter,
    # if basically creates a blank line
    self.info('')

    # Re-activate the real formatters
    for handler, formatter in zip(self.handlers, old_formatters):
        handler.setFormatter(formatter)
    

def log_message(self, level, message):
    """ Function to add to a logger object
    Log message, potentially with newlines to consider blank

    :param level: logging level ("info", "debug", "warning", "error")
    """
    # Creates as many blanklines after the message as needed
    while message[0] == "\n":
        self.newline()
        message = message[1:]

    # Call logger.info(message) or logger.warning(message)...
    getattr(self,level)(message)


def elapsed_time(self):
    """ Function to add to a logger object
    By calling logger.elpased_time(), we get the time in {}.{}s / {}mn{}.{}s / {}h{}mn{}.{}s
    since the begining of the logger creation
    """
    return nice_time(time.time()-self.t0)


def create_logger(name, file, verbose):
    """ Create a custom Python logger object, extends the logging library

    :param name: logger name, to access the appropriate logger if multiple of them are created
    :param file: logging file, will also log to stdout
    :param verbose: 0 -> logger will log only warning messages and above (error and critical)
                    anything else -> logger will log info messages and above (no debug)
    :returns: created logger
    """
    # Set appropriate logging level
    level = logging.INFO if verbose else logging.WARNING

    # Create custom logging config
    logging_config = {
        "version":1,
        "formatters": {
            "base-formatter": {
                "format": "[%(levelname)-8s] --- %(message)s" # Format logs as [level] --- message
            },
        },
        "handlers": {
            "{}-handler".format(name): { # Add file and level info to current logger name
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "base-formatter",
                "level": level,
                "filename": file
            },
            "console-handler": { # Handles the stdout stream config as well
                "class": "logging.StreamHandler",
                "formatter": "base-formatter",
                "level": level,
                "stream": sys.stdout
            },
        },
        "loggers": { # Add configs to logger
            name: {
                "handlers": ["{}-handler".format(name), "console-handler"],
                "level": level
            }
        }
    }

    # Activate defined config
    dictConfig(logging_config)

    # Create logger
    logger = logging.getLogger(name)
    # Define "newline" method to log blank lines, can be called with logger.newline()
    logger.newline = types.MethodType(log_blank, logger)
    # Define "write" method to log messages, can be called with logger.write(level, message)
    logger.write = types.MethodType(log_message, logger)
    # Define creation time
    logger.t0 = time.time()
    # Define "elapsed_time" method to get elapsed time since the creation of the logger
    logger.elapsed_time = types.MethodType(elapsed_time, logger)
    return logger


def get_relative_folder(input_folder, level):
    """ Get ancestor of a given folder
    :param input folder:
    :param level: relative ancestor level (add additional options as needed)
                  0 -> input folder
                  -1 -> parent folder
    :returns: ancestor path
    """
    if level == 0:
        return Path(input_folder)
    elif level == -1:
        return Path(input_folder).parent
    else:
        raise ValueError("Getting the relative parent level {} of a given folder is not supported yet.".format(level))


def load_csv(file, columns=[], dtype=None):
    """ Read .csv file if exists, otherwise create empty dataframe.
    :param file: file path (needs to end with .csv)
    :param columns: list of columns to create if file does not exist
    :param dtype: optional type for reading data from .csv
    :returns: Pandas dataframe 
    """
    if os.path.exists(file):
        if file.split(".")[-1] != "csv":
            raise ValueError("{} wrong format. Only .csv file types are supported for metadata information.".format(file))
        else:
            return pd.read_csv(file, dtype=dtype)
    else:
        return pd.DataFrame(columns=columns)


def nice_time(sec) :
    """ Nice time display
    :param sec: time in seconds (float)
    :returns: string {}.{}s / {}mn{}.{}s / {}h{}mn{}.{}s
    """
    sec_int, ms = str(sec).split(".")
    sec_int = int(sec_int)
    ms = ms[:4]
    mn_int = sec_int // 60
    s = sec_int - mn_int*60
    h = mn_int // 60
    mn = mn_int - h*60
    if h == 0:
        if mn == 0 :
            return "{}.{}s".format(s,ms)
        return "{}mn{}.{}s".format(mn,s,ms)
    return "{}h{}mn{}.{}s".format(h,mn,s,ms)


def get_files(input_folder, file_extension="", subdirs=False):
    """
    Get all files in specific folder

    :param input_folder: base folder in which to look for files
    :param file_extension: optional filter on the type of files to consider
    :param subdirs: set to True to look for files in subfolder depth levels >1
    :returns: list of file names (input_folder/[base file name])
    """
    if file_extension[0] != ".":
        file_extension = "." + file_extension
    
    # Only look for files present in input_folder
    if not subdirs:
        files = glob.glob(os.path.join(input_folder, "*"+file_extension))
    
    # Look for file in subfolders as well (all-path search)
    else:
        files = []
        # os.walk go through the entire subfolder structure, and outputs a list of tuple
        # (dir_path, subfolder_names, file_names)
        # where - 'dir_path' is the current subfolder
        #       - 'subfolder_names' is a list of all folders in 'dir_path'
        #       - 'file_names' is a list of all files in 'dir_path'
        for (dir_path, subfolder_names, file_names) in os.walk(input_folder):
            files += [
                os.path.join(dir_path, file_name) 
                for file_name in file_names 
                if file_extension in os.path.splitext(file_name)[1]
            ]
    
    return files


def get_basename(file, ext=False):
    ''' Return the basename of a file
    Input contain a full path

    :param file: file path
    :param ext: whether to return the extension(s) or not
    :returns: basename (/full/path/basename.ext), with or without extension
    '''
    if ext:
        return os.path.basename(file)
    else:
        return os.path.basename(file).split(".")[0]


def sort_from_reference(list_to_sort, reference_list, reverse=False):
    '''
    Sort a list based on the value of a second one
    Ex:
        - list_to_sort: ["red", "blue", "green", "yellow"]
        - reference_list: [10, 0, 6, 3]
        - result: ["blue", "yellow", "green", "red"]
    
    :param list_to_sort: primary list to be sorted
    :param reference_list: secondary list to use for sorting order
    :param reverse: set to True for descending order (default to False)
    :returns: sorted list_to_sort
    '''
    return [x for _, x in sorted(zip(reference_list, list_to_sort), key=lambda pair: pair[0], reverse=reverse)]
