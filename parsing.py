import re
import json

LOG_PATTERN = re.compile(
    r'^(?P<date>\d+)\s+(?P<time>\d+)\s+(?P<pid>\d+)\s+(?P<level>\w+)\s+(?P<component>[\w\.\$]+):\s+(?P<message>.*)$'
)


def parse_line(log_string):
    """
    Returns a dict, or None if the line does not match the expected log format.
    """
    match = LOG_PATTERN.match(log_string.strip())

    if not match:
        return None

    return match.groupdict()


def parse_file(file_path):
    """
    Parses a full log file into a list of structured dict, skipping malformed lines.
    """
    parsed_logs, skipped_lines = [], 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip():
                continue

            parsed_log = parse_line(line)

            if parsed_log:
                parsed_logs.append(parsed_log)

            else:
                skipped_lines += 1

    print(f"Parsed {len(parsed_logs)} lines, skipped {skipped_lines} malformed lines.")

    return parsed_logs


def save_json(parsed_logs, output_path):
    """
    Saves the parsed logs to a JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_logs, f, indent=2)


if __name__ == "__main__":
    logs = parse_file("HDFS_2k.log")
    save_json(logs, "parsed_logs.json")