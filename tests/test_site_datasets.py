import base64

import numpy as np

from src.experiments.site_datasets import _b64, _raw


def test_standardised_pixels_round_trip_to_grey_levels():
    grey = np.arange(256, dtype=np.uint8).reshape(16, 16)
    mean, std = 0.1307, 0.3081
    x = ((grey / 255.0 - mean) / (std + 1e-8)).astype(np.float32)
    np.testing.assert_array_equal(_raw(x, mean, std), grey)


def test_base64_packs_one_byte_per_pixel():
    a = np.arange(784, dtype=np.uint8)
    assert np.array_equal(np.frombuffer(base64.b64decode(_b64(a)), dtype=np.uint8), a)
