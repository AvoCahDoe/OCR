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


def test_plain_from_parsing_res_list():
    page = {
        "width": 482,
        "height": 284,
        "layout_det_res": {
            "boxes": [{"label": "text", "coordinate": [3, 18, 467, 278]}],
        },
        "parsing_res_list": [
            {"label": "text", "bbox": [3, 18, 467, 278], "content": "Article 2. – L'indemnisation"},
            {"label": "image", "content": ""},
        ],
    }
    assert _extract_plain(page) == "Article 2. – L'indemnisation"


def test_plain_from_printed_paddle_dump():
    dump = (
        "{'input_path': None, 'layout_det_res': {'boxes': []}, "
        "'parsing_res_list': [\n\n#################\nlabel:\ttext\n"
        "bbox:\t[3, 18, 467, 278]\ncontent:\tArticle 2. – L'indemnisation\n"
        "#################], 'dtype=uint8'}"
    )
    assert _extract_plain(dump) == "Article 2. – L'indemnisation"


def test_markdown_from_parsing_titles():
    page = {
        "parsing_res_list": [
            {"label": "paragraph_title", "content": "Title"},
            {"label": "text", "content": "Body"},
        ]
    }
    assert _extract_markdown(page) == "## Title\n\nBody"


def test_layout_omits_image_arrays():
    page = {
        "width": 10,
        "height": 20,
        "doc_preprocessor_res": {"output_img": "ndarray-not-here"},
        "layout_det_res": {
            "input_img": "nope",
            "boxes": [{"label": "text", "score": 0.9, "coordinate": [1, 2, 3, 4]}],
        },
        "parsing_res_list": [{"label": "text", "bbox": [1, 2, 3, 4], "content": "Hi"}],
    }
    layout = _extract_layout(page)
    assert layout["width"] == 10
    assert layout["blocks"][0]["content"] == "Hi"
    assert "output_img" not in layout
    assert "input_img" not in layout

