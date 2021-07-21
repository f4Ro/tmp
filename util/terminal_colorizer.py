class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def printc(*args: tuple, color: str = 'green') -> None:
    string = ''
    for arg in args:
        string += str(arg)
    print(_get_color_from_string(color) + string + bcolors.ENDC)


def _get_color_from_string(string: str) -> bcolors:
    return {
        'green': bcolors.OKGREEN,
        'blue': bcolors.OKBLUE,
        'cyan': bcolors.OKCYAN,
        'yellow': bcolors.WARNING,
        'red': bcolors.FAIL
    }[string.lower()]


if __name__ == '__main__':
    printc('test', 1)
    printc('a', 'abcnc ', 3, 3223, color=bcolors.BOLD)
