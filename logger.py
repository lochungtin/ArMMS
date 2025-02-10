from datetime import datetime
from os.path import join

from termcolor import colored


def _timestamp(format=False):
    rt = str(datetime.now()).split(".")[0]
    if format:
        return rt.replace("-", "_").replace(" ", "_").replace(":", "_")
    return rt


class Logger:
    def __init__(self, outDir):
        self.outDir = outDir
        self.log = open(join(outDir, "log.txt"), "w+")

    def _output(self, msg, toConsole, noNewLine, header="", color="white"):
        header = f"[{_timestamp()}] [{header}]" if header else f"[{_timestamp()}]"
        self.log.write(f"{header} - {msg}\n")
        if not noNewLine:
            self.log.write("\n")

        if toConsole:
            print(f"{colored(header, color)} - {msg}")

    def plain(self, msg, toConsole=True):
        self._output(msg, toConsole, True)

    def print(self, msg, toConsole=True, noNewLine=False):
        self._output(msg, toConsole, noNewLine)

    def action(self, msg, toConsole=True, noNewLine=False):
        self._output(msg, toConsole, noNewLine, header="Action", color="green")

    def warn(self, msg, toConsole=True, noNewLine=False):
        self._output(msg, toConsole, noNewLine, header="Warning", color="yellow")

    def error(self, msg, toConsole=True, noNewLine=False):
        self._output(msg, toConsole, noNewLine, header="ERROR", color="red")
        exit(1)

    def close(self):
        self.log.close()
