import re


class DataParser:
    def parse_analysis_content(self, content: str) -> dict:
        # Implement logic to parse the analysis content into structured data
        data = []
        for line in content.splitlines():
            match = re.search(r"(\w+): (\d+)", line)
            if match:
                data.append({"key": match.group(1), "value": int(match.group(2))})
        return {"data": data}

    def parse_history_content(self, content: str) -> dict:
        # Implement logic to parse the history content into structured data
        data = []
        for line in content.splitlines():
            match = re.search(r"(\w+): (\d{4}-\d{2}-\d{2})", line)
            if match:
                data.append({"event": match.group(1), "date": match.group(2)})
        return {"data": data}
