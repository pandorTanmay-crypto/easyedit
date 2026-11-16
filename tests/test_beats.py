"""Beat analysis cache identity."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from easyedit import beats


class BeatCacheTest(unittest.TestCase):
    def test_same_basename_different_sources_are_analyzed_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first" / "track.mp3"
            second = root / "second" / "track.mp3"
            first.parent.mkdir()
            second.parent.mkdir()
            first.touch()
            second.touch()
            cache = root / "work" / "beats-track.json"

            with mock.patch.object(beats, "load_audio", side_effect=[
                np.zeros(beats.SR * 2, dtype=np.float32),
                np.zeros(beats.SR * 3, dtype=np.float32),
            ]) as load_audio, \
                    mock.patch.object(beats, "onset_envelope", return_value=np.ones(10)), \
                    mock.patch.object(beats, "tempo", return_value=120.0), \
                    mock.patch.object(beats, "track", return_value=np.array([0.0, 0.5, 1.0, 1.5])):
                beats.analyze(first, cache)
                result = beats.analyze(second, cache)

            self.assertEqual(load_audio.call_count, 2)
            self.assertEqual(load_audio.call_args_list[1].args[0], second)
            self.assertEqual(result["source"], str(second.resolve()))
            self.assertEqual(result["duration"], 3.0)

    def test_unchanged_source_reuses_its_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "track.mp3"
            source.touch()
            cache = root / "beats-track.json"
            audio = np.zeros(beats.SR * 2, dtype=np.float32)

            with mock.patch.object(beats, "load_audio", return_value=audio) as load_audio, \
                    mock.patch.object(beats, "onset_envelope", return_value=np.ones(10)), \
                    mock.patch.object(beats, "tempo", return_value=120.0), \
                    mock.patch.object(beats, "track", return_value=np.array([0.0, 0.5, 1.0, 1.5])):
                first = beats.analyze(source, cache)
                second = beats.analyze(source, cache)

            load_audio.assert_called_once_with(source)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
