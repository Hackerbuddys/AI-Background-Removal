import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import cv2
import io
import time
from rembg import remove, new_session

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Background Removal System Pro",
    layout="wide",
    page_icon="🧠"
)

st.title("🖼 AI-Powered Background Removal System (Pro Version)")

# ---------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------

st.sidebar.header("⚙ Settings")

image_type = st.sidebar.selectbox(
    "Image Type",
    ["Logo / Text Image", "Human Photo", "Product Image"]
)

edge_smooth = st.sidebar.slider("Edge Smoothness", 1, 9, 3, step=2)

replace_bg = st.sidebar.checkbox("Replace Background")

replace_mode = st.sidebar.selectbox(
    "Background Mode",
    ["Solid Color", "Custom Image"]
)

bg_color = st.sidebar.color_picker("Background Color", "#FFFFFF")

bg_upload = None
if replace_mode == "Custom Image":
    bg_upload = st.sidebar.file_uploader("Upload Background Image", type=["png", "jpg", "jpeg"])

blur_bg = st.sidebar.checkbox("Blur Background")

auto_crop = st.sidebar.checkbox("Auto Crop Subject")

add_shadow = st.sidebar.checkbox("Add Drop Shadow")

download_format = st.sidebar.selectbox("Download Format", ["PNG", "JPG", "WEBP"])
download_quality = st.sidebar.slider("Download Quality", 70, 100, 100)

# ---------------------------------------------------
# AI SESSION CACHE (LIGHTWEIGHT MODEL)
# ---------------------------------------------------

@st.cache_resource
def load_session(model):
    return new_session(model)

# ---------------------------------------------------
# LOGO MODE (WHITE REMOVAL)
# ---------------------------------------------------

def remove_white_background(img, threshold=245):
    img = img.convert("RGBA")
    np_img = np.array(img)

    gray = cv2.cvtColor(np_img[:, :, :3], cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    np_img[:, :, 3] = mask
    return Image.fromarray(np_img)

# ---------------------------------------------------
# AI MODE (USING LIGHTWEIGHT u2netp)
# ---------------------------------------------------

def remove_ai_background(img, model_name):
    session = load_session(model_name)

    output = remove(
        img,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=200,
        alpha_matting_background_threshold=5,
        alpha_matting_erode_structure_size=3
    )
    return output

# ---------------------------------------------------
# AUTO CROP
# ---------------------------------------------------

def crop_transparent(image):
    bbox = image.getbbox()
    if bbox:
        return image.crop(bbox)
    return image

# ---------------------------------------------------
# SHADOW EFFECT
# ---------------------------------------------------

def add_shadow_effect(image):
    shadow = image.copy().convert("RGBA")
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    bg = Image.new("RGBA", image.size, (0, 0, 0, 0))
    bg.paste(shadow, (10, 10), shadow)
    bg.paste(image, (0, 0), image)
    return bg

# ---------------------------------------------------
# BACKGROUND REPLACEMENT
# ---------------------------------------------------

def replace_with_color(fg, hex_color):
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    bg = Image.new("RGBA", fg.size, rgb + (255,))
    return Image.alpha_composite(bg, fg).convert("RGB")

def replace_with_image(fg, bg_img):
    bg = bg_img.resize(fg.size).convert("RGBA")
    return Image.alpha_composite(bg, fg).convert("RGB")

# ---------------------------------------------------
# DOWNLOAD HANDLER
# ---------------------------------------------------

def export_image(image, fmt, quality):
    buf = io.BytesIO()
    if fmt == "PNG":
        image.save(buf, format="PNG")
    elif fmt == "JPG":
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
    elif fmt == "WEBP":
        image.save(buf, format="WEBP", quality=quality)
    return buf.getvalue()

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    original = Image.open(uploaded_file).convert("RGBA")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original, use_container_width=True)

    if st.button("✨ Remove Background"):

        start = time.time()

        with st.spinner("Processing..."):

            if image_type == "Logo / Text Image":
                processed = remove_white_background(original)
            elif image_type == "Human Photo":
                processed = remove_ai_background(original, "u2netp")
            else:
                processed = remove_ai_background(original, "u2netp")

            if auto_crop:
                processed = crop_transparent(processed)

            if add_shadow:
                processed = add_shadow_effect(processed)

            if replace_bg:
                if replace_mode == "Solid Color":
                    processed = replace_with_color(processed, bg_color)
                elif replace_mode == "Custom Image" and bg_upload:
                    bg_img = Image.open(bg_upload)
                    if blur_bg:
                        bg_img = bg_img.filter(ImageFilter.GaussianBlur(8))
                    processed = replace_with_image(processed, bg_img)

        end = time.time()

        with col2:
            st.subheader("Processed Image")
            st.image(processed, use_container_width=True)

        st.success(f"Processing Time: {round(end - start, 2)} seconds")

        file_bytes = export_image(processed, download_format, download_quality)

        st.download_button(
            label=f"⬇ Download {download_format}",
            data=file_bytes,
            file_name=f"output.{download_format.lower()}",
            mime=f"image/{download_format.lower()}"
        )

else:
    st.info("Upload an image to start.")
