import time as pytime

from termcolor import colored


def warning(text: str):
    print(colored(text, "yellow"))


def info(text: str):
    print(colored(text, "blue"))


def success(text: str):
    print(colored(text, "green"))


def time(text: str) -> "TimeLogger":
    return TimeLogger(text)


class TimeLogger:
    def __init__(self, text: str):
        self.text = text
        self.start_time = pytime.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = pytime.time()
        elapsed_ms = (end_time - self.start_time) * 1000
        print(colored(f"{self.text} in {elapsed_ms:.2f}ms", "cyan"))
