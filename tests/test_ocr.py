from __future__ import annotations

from ocr import _extract_layout, _extract_markdown, _extract_plain, _markdown_to_plain


class FakePage:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_plain_from_text_attr():
    assert _extract_plain(FakePage(text="hello\n\n\nworld")) == "hello\n\nworld"


def test_plain_from_markdown_fallback():
    page = FakePage(markdown="# Title\n\nSome **bold** text")
    assert "Title" in _extract_plain(page)
    assert "**" not in _extract_plain(page)


def test_markdown_from_dict():
    page = {"markdown": {"text": "# Hi"}}
    assert _extract_markdown(page) == "# Hi"


def test_layout_from_json_attr():
    page = FakePage(json={"boxes": [{"label": "text"}]})
    assert _extract_layout(page)["boxes"][0]["label"] == "text"


def test_markdown_to_plain():
    md = "# Head\n\nA [link](http://x) and `code`"
    plain = _markdown_to_plain(md)
    assert "Head" in plain
    assert "link" in plain
    assert "http" not in plain
