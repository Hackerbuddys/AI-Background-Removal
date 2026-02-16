from PIL import Image

def add_background(foreground, background_color=None, background_image=None):
    """
    Replace transparent background.
    """

    fg = foreground.convert("RGBA")

    if background_image:
        bg = background_image.resize(fg.size).convert("RGBA")
    else:
        bg = Image.new("RGBA", fg.size, background_color or (255, 255, 255, 255))

    combined = Image.alpha_composite(bg, fg)

    return combined.convert("RGB")
