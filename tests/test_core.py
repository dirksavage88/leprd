import unittest

import numpy as np

from ekf2_playback_review.analysis import bool_spans, format_spans


class SpanTests(unittest.TestCase):
    def test_bool_spans_empty(self):
        self.assertEqual(bool_spans(np.array([]), np.array([])), [])

    def test_bool_spans_segments(self):
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([False, True, True, False, True])
        self.assertEqual(bool_spans(t, values), [(1.0, 2.0), (4.0, 4.0)])

    def test_format_spans(self):
        self.assertEqual(format_spans([]), "none")
        self.assertEqual(format_spans([(1.0, 2.0), (4.0, 4.0)]), "1.000--2.000s, 4.000s")


if __name__ == "__main__":
    unittest.main()
