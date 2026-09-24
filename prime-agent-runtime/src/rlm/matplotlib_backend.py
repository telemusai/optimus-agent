"""Headless Matplotlib backend for inline Optimus image attachments.

Loaded by Matplotlib on first use, never during ordinary kernel startup.
"""

from __future__ import annotations

import base64
import io
import math

import matplotlib
from matplotlib import _pylab_helpers
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

_ATTACHMENT_MIME = "application/vnd.prime-agent.attachment+json"
_MAX_DIMENSION = 1200
_MAX_DATA_CHARS = 350_000


def _figure_png(figure) -> bytes:
    width, height = figure.get_size_inches()
    if not all(math.isfinite(value) and value > 0 for value in (width, height, figure.dpi)):
        raise ValueError("Cannot preview a figure with invalid dimensions or DPI")
    dpi = min(figure.dpi, _MAX_DIMENSION / max(width, height))
    buffer = io.BytesIO()
    # A global tight-bbox setting must not defeat the bounded raster size.
    with matplotlib.rc_context({"savefig.bbox": None}):
        figure.savefig(buffer, format="png", dpi=dpi)
    data = buffer.getvalue()
    source = Image.open(io.BytesIO(data))
    try:
        while ((len(data) + 2) // 3) * 4 > _MAX_DATA_CHARS:
            size = (max(1, int(source.width * 0.75)), max(1, int(source.height * 0.75)))
            resized = source.resize(size, Image.Resampling.LANCZOS)
            source.close()
            source = resized
            buffer = io.BytesIO()
            source.save(buffer, format="PNG")
            data = buffer.getvalue()
    finally:
        source.close()
    return data


class FigureManager(FigureManagerBase):
    _previewed = False

    def show(self):
        from .repl import emit

        data = _figure_png(self.canvas.figure)
        emit({
            _ATTACHMENT_MIME: {"mime_type": "image/png", "data": base64.b64encode(data).decode("ascii")},
            "text/plain": f"Matplotlib figure {self.num}: inline image preview",
        })
        self._previewed = True
        self.canvas.figure.stale = False

    @classmethod
    def pyplot_show(cls, *, block=None):
        for manager in _pylab_helpers.Gcf.get_all_fig_managers():
            manager.show()


class FigureCanvas(FigureCanvasAgg):
    manager_class = FigureManager


def flush_figures() -> None:
    """Preview new or changed figures once at the end of a successful cell."""
    for manager in _pylab_helpers.Gcf.get_all_fig_managers():
        if isinstance(manager, FigureManager) and (not manager._previewed or manager.canvas.figure.stale):
            manager.show()
