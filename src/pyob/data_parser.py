import re


class DataParser:
    def parse_analysis_content(self, content: str) -> dict:
        """
        Safely parses numeric stats from analysis content while stripping CSS units
        to prevent 'invalid decimal literal' errors.
        """
        data = []
        for line in content.splitlines():
            # Skip complex CSS calc() functions that confuse the parser
            if "calc(" in line:
                continue

            # This regex captures the label and the number, then ignores optional units
            match = re.search(r"(\w+)\s*:\s*([\d\.]+)(?:px|em|rem|%|s)?", line)
            if match:
                key = match.group(1)
                value_str = match.group(2)
                try:
                    # Convert to float if there is a decimal, otherwise int
                    value = float(value_str) if "." in value_str else int(value_str)
                    data.append({"key": key, "value": value})
                except ValueError:
                    # If conversion fails for any reason, skip this line safely
                    continue
        return {"data": data}

    def parse_history_content(self, content: str) -> dict:
        """Parses event history into structured data."""
        data = []
        for line in content.splitlines():
            # Matches 'EventName: YYYY-MM-DD'
            match = re.search(r"(\w+): (\d{4}-\d{2}-\d{2})", line)
            if match:
                data.append({"event": match.group(1), "date": match.group(2)})
        return {"data": data}
