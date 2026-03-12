import os

from pyob.feature_mixins import FeatureOperationsMixin


# 1. Create a dummy class that implements the Mixin so we can test it independently
class DummyReviewer(FeatureOperationsMixin):
    def __init__(self, target_dir):
        self.target_dir = target_dir
        self.pr_file = os.path.join(target_dir, "PEER_REVIEW.md")
        self.session_context = []


def test_write_pr_creates_and_formats_file(tmp_path):
    """
    Test that the write_pr method correctly creates the PEER_REVIEW.md
    file and formats the AI's explanation and XML edits properly.
    """
    # tmp_path is a built-in pytest fixture that creates a temporary directory for safe testing
    reviewer = DummyReviewer(str(tmp_path))

    # Fake data to simulate the AI's response
    test_filepath = os.path.join(str(tmp_path), "fake_module.py")
    explanation = "Fixed a missing import."
    llm_response = (
        "<EDIT>\n"
        "<SEARCH>\nprint('hello')\n</SEARCH>\n"
        "<REPLACE>\nimport sys\nprint('hello')\n</REPLACE>\n"
        "</EDIT>"
    )

    # Run the method we are testing
    reviewer.write_pr(test_filepath, explanation, llm_response)

    # ASSERTIONS: Verify the method did exactly what it was supposed to do
    assert os.path.exists(reviewer.pr_file), "PEER_REVIEW.md was not created!"

    with open(reviewer.pr_file, "r", encoding="utf-8") as f:
        content = f.read()

        # Check that the file contains the right markdown headers and our fake data
        assert "Autonomous Code Review" in content
        assert "Fixed a missing import." in content
        assert "```xml" in content
        assert "<SEARCH>" in content
        assert "import sys" in content
