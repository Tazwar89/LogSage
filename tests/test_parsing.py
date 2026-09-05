"""
Tests for app/parsing.py

Covers: well-formed HDFS-style log lines, malformed lines, empty input,
and the file-level batch parser skipping bad lines without crashing.
"""
import json
import pytest

from app.parsing import parse_line, parse_file, save_json


class TestParseLine:
    def test_parses_well_formed_line(self):
        line = "081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating"
        result = parse_line(line)

        assert result is not None
        assert result["date"] == "081109"
        assert result["time"] == "203615"
        assert result["pid"] == "148"
        assert result["level"] == "INFO"
        assert result["component"] == "dfs.DataNode$PacketResponder"
        assert result["message"] == "PacketResponder 1 for block blk_38865049064139660 terminating"

    def test_returns_none_for_malformed_line(self):
        assert parse_line("this is not a valid log line") is None

    def test_returns_none_for_empty_string(self):
        assert parse_line("") is None

    def test_returns_none_for_missing_fields(self):
        # Missing PID and level
        assert parse_line("081109 203615 dfs.DataNode: something happened") is None

    def test_handles_trailing_whitespace_and_carriage_return(self):
        line = "081109 203615 148 INFO dfs.DataNode$PacketResponder: terminating\r\n"
        result = parse_line(line)

        assert result is not None
        assert result["message"] == "terminating"

    def test_component_allows_dollar_sign_and_dots(self):
        line = "081109 203615 148 INFO dfs.FSNamesystem$Something.Nested: msg here"
        result = parse_line(line)

        assert result is not None
        assert result["component"] == "dfs.FSNamesystem$Something.Nested"


class TestParseFile:
    def test_parses_all_well_formed_lines(self, tmp_path):
        log_file = tmp_path / "sample.log"
        log_file.write_text(
            "081109 203615 148 INFO dfs.DataNode$PacketResponder: line one\n"
            "081109 203616 149 INFO dfs.DataNode$PacketResponder: line two\n"
        )

        parsed = parse_file(str(log_file))

        assert len(parsed) == 2
        assert parsed[0]["message"] == "line one"
        assert parsed[1]["message"] == "line two"

    def test_skips_malformed_lines_without_crashing(self, tmp_path):
        log_file = tmp_path / "mixed.log"
        log_file.write_text(
            "081109 203615 148 INFO dfs.DataNode$PacketResponder: good line\n"
            "not a valid log line at all\n"
            "081109 203617 150 INFO dfs.DataNode$PacketResponder: another good line\n"
        )

        parsed = parse_file(str(log_file))

        assert len(parsed) == 2
        assert all("message" in entry for entry in parsed)

    def test_skips_blank_lines(self, tmp_path):
        log_file = tmp_path / "blanks.log"
        log_file.write_text(
            "081109 203615 148 INFO dfs.DataNode$PacketResponder: good line\n"
            "\n"
            "   \n"
            "081109 203617 150 INFO dfs.DataNode$PacketResponder: another good line\n"
        )

        parsed = parse_file(str(log_file))

        assert len(parsed) == 2

    def test_empty_file_returns_empty_list(self, tmp_path):
        log_file = tmp_path / "empty.log"
        log_file.write_text("")

        parsed = parse_file(str(log_file))

        assert parsed == []


class TestSaveJson:
    def test_writes_valid_json(self, tmp_path):
        out_file = tmp_path / "out.json"
        data = [{"message": "test"}]

        save_json(data, str(out_file))

        with open(out_file) as f:
            loaded = json.load(f)

        assert loaded == data
