"""
Example driver script to be used
"""

import argparse
import sys


JOBNAME = "MLFF Inference"


class ExampleDriver:
    def __init__(self, options, jobname):
        self.options = options
        self.jobname = jobname

    def run(self):
        self.initVariables()
        self.doCalculation()
        self.doMoreCalculation()
        self.exportData()

    def initVariables(self):
        pass

    def doCalculation(self):
        pass

    def doMoreCalculation(self):
        pass

    def exportData(self):
        pass


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    # Add arguments to the parser as needed
    return parser


def validate_options(options, parser):
    pass


def main(args):
    parser = get_parser()
    options = parser.parse_args(args)
    validate_options(options, parser)

    driver = ExampleDriver(options, JOBNAME)
    driver.run()


if __name__ == "__main__":
    main(sys.argv[1:])
