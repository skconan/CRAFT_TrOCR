import pytest

from utils.preprocessor import read_media


def test_read_png():
    img_list = read_media(r"data\images\test_00.png")
    assert len(img_list) == 1
    assert img_list[0].shape == (850, 635, 3)


def test_read_pdf():
    img_list = read_media(r"data\pdf\test_00.pdf")
    assert len(img_list) == 2
    assert img_list[0].shape == (2339, 1656, 3)
