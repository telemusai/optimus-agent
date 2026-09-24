---
name: attach-image
description: Show an on-disk image (PNG, JPEG, GIF, WebP) in the chat and load it into the model's context as a viewable attachment. Use for screenshots, diagrams, charts, photos, or scanned pages, including when the user asks to see an image or show it again. Requires a vision-capable model; errors clearly otherwise.
---

# Attach Image

Load on-disk images into the model's context as multimodal attachments. The
image is sent to the model the same way a pasted image is, so the model can
actually look at it.
The same attachment supplies the inline chat preview when terminal image display
is enabled, which is the default.

## When to use this

- The user points at an image file and wants you to look at it.
- You need to read text, a chart, a diagram, or a layout from an image.
- A screenshot needs visual interpretation.
- The user asks you to show a screenshot, chart, or other image in the chat.
- The user asks to see an image again: emit a fresh attachment in that turn.

## When NOT to use this

For *programmatic* work on an image — measuring pixels, cropping, resizing,
computing a hash, comparing files byte-by-byte — open it in the kernel with a
library instead:

```python
from PIL import Image
img = Image.open("diagram.png")
print(img.size)
```

That path does not put the image in the model's context; it only lets you
compute over it. Use `attach_image` when you need to *see* the image.

## Usage

Call the prepared `attach_image` import directly in the Python kernel:

```python
print(await attach_image("diagram.png"))
print(await attach_image("a.png", "b.jpg"))
```

For screenshots or generated charts, wait for the capture or generation command
to finish successfully before attaching the file. After the attachment succeeds,
a short caption is enough. Local Markdown image links such as
`![Screenshot](/tmp/shot.png)` only render as text links in the terminal; they do
not display an image. Printing a path or returning a file link is also insufficient.
If the user asks to see it again, call `attach_image` again. If attaching fails,
report the error and resolve it before claiming the image is shown.

Matplotlib figures created in the Python workspace preview automatically when a
successful cell finishes. Use `plt.show()` or `fig.show()` to show them explicitly
or again. Keep the default inline backend for this; explicitly selecting `Agg`
or another backend opts out. Use `attach_image` for an image saved by a separate
script or for an existing file.

The skill automatically resizes and compresses large images before loading them
into context. Animated images that need compression are flattened to their first
frame. Transparent images that need compression are composited onto a neutral
gray background. Extremely large images are rejected by pixel count before full
processing. The original file is left untouched.

Supported formats: PNG, JPEG, GIF, WebP. The skill errors if a file is not a
supported image, or if the current model is not vision-capable.
