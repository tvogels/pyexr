import pytest

import pyexr


@pytest.fixture
def data():
    return 1


def test_dummy(data):
    assert data == 1
