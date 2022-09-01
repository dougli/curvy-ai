from termcolor import colored


def warning(text: str):
    print(colored(text, "yellow"))


def info(text: str):
    print(colored(text, "blue"))


def success(text: str):
    print(colored(text, "green"))
