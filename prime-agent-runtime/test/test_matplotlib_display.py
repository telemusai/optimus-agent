from __future__ import annotations

import base64
import importlib.util
import io
import tempfile
import unittest

from test_repl import ReplProcess, one


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is optional")
class MatplotlibDisplayTest(unittest.TestCase):
    def setUp(self):
        self.config = tempfile.TemporaryDirectory()
        self.addCleanup(self.config.cleanup)
        self.repl = ReplProcess({"MPLBACKEND": "", "MPLCONFIGDIR": self.config.name})
        self.addCleanup(self.repl.close)
        self.repl.ready()

    def images(self, events):
        self.assertEqual(one(events, "done")["status"], "ok", events)
        return [event["data"]["application/vnd.prime-agent.attachment+json"]
                for event in events if event.get("event") == "display"]

    def check_png(self, image):
        from PIL import Image

        self.assertEqual(image["mime_type"], "image/png")
        self.assertLessEqual(len(image["data"]), 350_000)
        with Image.open(io.BytesIO(base64.b64decode(image["data"], validate=True))) as png:
            png.load()
            self.assertLessEqual(max(png.size), 1200)
            # The background uses the same neon canvas as Optimus.
            self.assertEqual(png.convert("RGB").getpixel((0, 0)), (8, 13, 12))

    def test_plotting_cell_previews_once_and_changed_figures_preview_again(self):
        events = self.repl.execute("plot", """
import matplotlib.pyplot as plt
fig, ax = plt.subplots(facecolor='#080d0c')
ax.plot([0, 1, 2], [0, 3, 1], color='#00f477')
""")
        images = self.images(events)
        self.assertEqual(len(images), 1)
        self.check_png(images[0])
        self.assertTrue(all(event.get("id") == "plot" for event in events if event.get("event") == "display"))
        self.assertEqual(self.images(self.repl.execute("unchanged", "1 + 1")), [])
        self.assertEqual(len(self.images(self.repl.execute("changed", "ax.set_title('Updated')"))), 1)
        self.assertEqual(self.images(self.repl.execute("close", "plt.close(fig)")), [])

    def test_show_can_repeat_without_duplicate_end_of_cell_output(self):
        events = self.repl.execute("show", """
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20, 12), dpi=300, facecolor='#080d0c')
ax.plot([1, 2], [3, 4], color='#e201ea')
plt.show()
""")
        images = self.images(events)
        self.assertEqual(len(images), 1)
        self.check_png(images[0])
        self.assertEqual(len(self.images(self.repl.execute("again", "plt.show()"))), 1)
        self.assertEqual(len(self.images(self.repl.execute("figure-show", "fig.show()"))), 1)
        state = self.repl.execute("original-size", "(float(fig.dpi), tuple(map(float, fig.get_size_inches())))")
        self.assertEqual(one(state, "result")["text"], "(300.0, (20.0, 12.0))")
        self.assertEqual(self.images(state), [])

    def test_dense_chart_is_bounded_for_chat_replay(self):
        events = self.repl.execute("dense", """
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 12), dpi=100, facecolor='#080d0c')
ax.imshow(np.random.default_rng(7).integers(0, 256, (1200, 1200, 3), dtype=np.uint8))
""")
        images = self.images(events)
        self.assertEqual(len(images), 1)
        self.check_png(images[0])

    def test_backend_does_not_import_matplotlib_during_startup(self):
        events = self.repl.execute("lazy", "import sys; 'matplotlib' in sys.modules")
        self.assertEqual(one(events, "result")["text"], "False")

    def test_explicit_backend_is_respected(self):
        repl = ReplProcess({"MPLBACKEND": "Agg", "MPLCONFIGDIR": self.config.name})
        self.addCleanup(repl.close)
        repl.ready()
        events = repl.execute("agg", "import matplotlib.pyplot as plt; plt.plot([1, 2]); plt.get_backend()")
        self.assertEqual(self.images(events), [])
        self.assertEqual(one(events, "result")["text"].lower(), "'agg'")


if __name__ == "__main__":
    unittest.main()
