import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import time
from rembg import remove, new_session

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Background Removal System",
    layout="wide",
    page_icon="🧠"
)

# ---------------------------------------------------
# CUSTOM THEME
# ---------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 0.5em 1em;
}
.stButton>button:hover {
    background-color: #1e40af;
}
</style>
""", unsafe_allow_html=True)

st.title("🖼 AI-Powered Background Removal System")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("⚙ Settings")

image_type = st.sidebar.selectbox(
    "Image Type",
    ["Logo / Text Image", "Human Photo", "Product Image"]
)

edge_smooth = st.sidebar.slider("Edge Smoothness", 1, 9, 3, step=2)

replace_bg = st.sidebar.checkbox("Replace Background")

bg_color = st.sidebar.color_picker("Background Color", "#FFFFFF")

# ---------------------------------------------------
# AI SESSION CACHE
# ---------------------------------------------------

_sessions = {}

def get_session(model):
    if model not in _sessions:
        _sessions[model] = new_session(model)
    return _sessions[model]

# ---------------------------------------------------
# LOGO MODE (COLOR BASED REMOVAL)
# ---------------------------------------------------

def remove_white_background(img, threshold=245):
    img = img.convert("RGBA")
    np_img = np.array(img)

    gray = cv2.cvtColor(np_img[:, :, :3], cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_not(mask)

    # Smooth mask edges
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    np_img[:, :, 3] = mask

    return Image.fromarray(np_img)

# ---------------------------------------------------
# AI MODE (PHOTO SEGMENTATION)
# ---------------------------------------------------

def remove_ai_background(img, model_name="u2net"):
    session = get_session(model_name)

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
# OPTIONAL BACKGROUND REPLACEMENT
# ---------------------------------------------------

def add_background(foreground, hex_color):
    fg = foreground.convert("RGBA")

    rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

    bg = Image.new("RGBA", fg.size, rgb + (255,))
    combined = Image.alpha_composite(bg, fg)

    return combined.convert("RGB")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGBA")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(img, use_container_width=True)

    if st.button("✨ Remove Background"):

        start = time.time()

        with st.spinner("Processing..."):

            if image_type == "Logo / Text Image":
                processed = remove_white_background(img)
            elif image_type == "Human Photo":
                processed = remove_ai_background(img, "u2net_human_seg")
            else:
                processed = remove_ai_background(img, "u2net")

            if replace_bg:
                processed = add_background(processed, bg_color)

        end = time.time()

        with col2:
            st.subheader("Processed")
            st.image(processed, use_container_width=True)

        st.success(f"Processing Time: {round(end - start, 2)} seconds")

        # Download
        buf = io.BytesIO()
        processed.save(buf, format="PNG")

        st.download_button(
            label="⬇ Download Image",
            data=buf.getvalue(),
            file_name="background_removed.png",
            mime="image/png"
        )

else:
    st.info("Upload an image to start.")
