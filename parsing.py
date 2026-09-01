import re

def parse_input(log_string):
    """
    Parses the log string and returns a list tokenized as date, time, pid, level, component, and message.

    Args:
        log_string (str): The log string to be parsed.

    Returns:
        list: A list of tokens.
    """
    date = log_string.split()[0]
    time = log_string.split()[1]
    pid = log_string.split()[2]
    level = log_string.split()[3]
    component = log_string.split()[4].strip(':')
    message = ' '.join(log_string.split()[5:])

    return [date, time, pid, level, component, message]

string = "081109 204005 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.addStoredBlock:" \
"blockMap updated: 10.251.73.220:50010 is added to blk_7128370237687728475 size 67108864"
print(parse_input(string))