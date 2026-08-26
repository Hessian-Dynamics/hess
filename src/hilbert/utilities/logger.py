"""
Logger module implemented using Python's native logging library.
"""

import datetime
import logging


def get_logfile_name(jobname):
    """
    Returns a clean log file name based on the jobname.
    """
    clean_filename = jobname.replace(" ", "_").replace("/", "_")
    return f"{clean_filename}.log"


def TextLogger(jobname="Job"):
    """
    Factory function using built-in logging module to write to jobname.log.
    """

    log_filename = get_logfile_name(jobname)

    log_obj = logging.getLogger(jobname)
    log_obj.setLevel(logging.INFO)

    if not log_obj.handlers:
        file_handler = logging.FileHandler(
            log_filename, mode="a", encoding="utf-8"
        )
        log_obj.addHandler(file_handler)

    def log(message, pad=False, pad_below=False, timestamp=False):
        prefix = ""
        if timestamp:
            now_str = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
            prefix = now_str

        formatted_message = f"{prefix}{message}"

        if pad:
            for h in log_obj.handlers:
                h.stream.write("\n")

        log_obj.info(formatted_message)

        if pad_below:
            for h in log_obj.handlers:
                h.stream.write("\n")

    return log
