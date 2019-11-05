# -*- coding: utf-8 -*-

import os
import pytest


@pytest.fixture(scope="module")
def setup_output_dir_for_fashion_tests():
    test_output_dir = 'tests/output/fashion'
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)
