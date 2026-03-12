from pyob.apply_xml_mixins import ApplyXMLMixin


def test_apply_xml_edits_precision():
    mixin = ApplyXMLMixin()
    
    source = "def add(a, b):\n    return a + b\n"
    llm_response = """
<THOUGHT>Fixing logic</THOUGHT>
<EDIT>
<SEARCH>
    return a + b
</SEARCH>
<REPLACE>
    # Logic Updated
    return (a + b)
</REPLACE>
</EDIT>
"""
    new_code, explanation, success = mixin.apply_xml_edits(source, llm_response)

    assert success is True
    assert "# Logic Updated" in new_code
    assert "return (a + b)" in new_code
    assert explanation == "Fixing logic"
