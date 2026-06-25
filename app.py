import streamlit as st
from PIL import Image
import numpy as np
import cv2
import time
import torch
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="VisionAI Studio",
    page_icon="🔮",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    font-size: 3rem;
    font-weight: 700;
    color: white;
}

.subtitle {
    color: #A0A0A0;
    font-size: 1.1rem;
    margin-bottom: 20px;
}

.metric-card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #30363D;
}

.metric-value {
    font-size: 2rem;
    color: #58A6FF;
    font-weight: bold;
}

.metric-label {
    color: #8B949E;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------

@st.cache_resource
def load_models():

    # Load SAM
    sam_checkpoint = "models/sam_vit_b_01ec64.pth"

    sam = sam_model_registry["vit_b"](
        checkpoint=sam_checkpoint
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam.to(device=device)

    predictor = SamPredictor(sam)

    # Load YOLO
    yolo_model = YOLO("yolov8n.pt")

    return predictor, yolo_model


predictor, yolo_model = load_models()

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<div class="title">🔮 VisionAI Segmentation Studio</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">YOLOv8 + Meta SAM Powered Intelligent Segmentation System</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("⚙️ Controls")

mode = st.sidebar.selectbox(
    "Select Mode",
    [
        "Segmentation Overlay",
        "Blur Background",
        "Remove Background"
    ]
)

blur_strength = 35

if mode == "Blur Background":

    blur_strength = st.sidebar.slider(
        "Blur Strength",
        5,
        99,
        35,
        step=2
    )

device_name = (
    "GPU CUDA 🚀"
    if torch.cuda.is_available()
    else "CPU 💻"
)

st.sidebar.info(f"Running on: {device_name}")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    # ---------------------------------------------------
    # PROCESS BUTTON
    # ---------------------------------------------------

    if st.button("🚀 Run AI Segmentation"):

        start_time = time.time()

        with st.spinner("Running YOLO + SAM Pipeline..."):

            # -------------------------------------------
            # YOLO DETECTION
            # -------------------------------------------

            results = yolo_model(image_np)

            boxes = results[0].boxes.xyxy.cpu().numpy()

            confidences = results[0].boxes.conf.cpu().numpy()

            class_ids = results[0].boxes.cls.cpu().numpy()

            names = results[0].names

            # -------------------------------------------
            # SAM PREPARATION
            # -------------------------------------------

            predictor.set_image(image_np)

            segmented_objects = []

            combined_mask = np.zeros(
                image_np.shape[:2],
                dtype=bool
            )

            result_image = image_np.copy()

            # -------------------------------------------
            # SEGMENT EACH OBJECT
            # -------------------------------------------

            for box, conf, cls_id in zip(
                boxes,
                confidences,
                class_ids
            ):

                x1, y1, x2, y2 = box.astype(int)

                input_box = np.array([x1, y1, x2, y2])

                masks, scores, logits = predictor.predict(
                    box=input_box,
                    multimask_output=False
                )

                mask = masks[0]

                combined_mask |= mask

                segmented_objects.append({
                    "box": input_box,
                    "mask": mask,
                    "confidence": conf,
                    "class": names[int(cls_id)]
                })

                # RANDOM COLOR MASK
                color = np.random.randint(
                    0,
                    255,
                    (3,)
                )

                # OVERLAY
                result_image[mask] = (
                    result_image[mask] * 0.5
                    + color * 0.5
                )

                # DRAW BOX
                cv2.rectangle(
                    result_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # LABEL
                label = f"{names[int(cls_id)]} {conf:.2f}"

                cv2.putText(
                    result_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

            # -------------------------------------------
            # APPLY MODES
            # -------------------------------------------

            if mode == "Blur Background":

                blurred = cv2.GaussianBlur(
                    image_np,
                    (blur_strength, blur_strength),
                    0
                )

                result_image = np.where(
                    combined_mask[:, :, None],
                    image_np,
                    blurred
                )

            elif mode == "Remove Background":

                rgba = cv2.cvtColor(
                    image_np,
                    cv2.COLOR_RGB2RGBA
                )

                rgba[~combined_mask, 3] = 0

                result_image = rgba

            # -------------------------------------------
            # PROCESS TIME
            # -------------------------------------------

            end_time = time.time()

            processing_time = round(
                end_time - start_time,
                2
            )

            object_count = len(segmented_objects)

        # ---------------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📷 Original Image")

            st.image(
                image_np,
                use_container_width=True
            )

        with col2:

            st.subheader("✨ AI Processed Output")

            st.image(
                result_image,
                use_container_width=True
            )

        # ---------------------------------------------------
        # METRICS
        # ---------------------------------------------------

        st.markdown("---")

        m1, m2, m3 = st.columns(3)

        with m1:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{object_count}</div>
                <div class="metric-label">Objects Detected</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{processing_time}s</div>
                <div class="metric-label">Processing Time</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:

            avg_conf = np.mean(confidences) * 100

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_conf:.1f}%</div>
                <div class="metric-label">Average Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        # ---------------------------------------------------
        # OBJECT DETAILS
        # ---------------------------------------------------

        st.markdown("---")

        st.subheader("📊 Detection Details")

        for obj in segmented_objects:

            st.write(
                f"• {obj['class']} "
                f"(Confidence: {obj['confidence']:.2f})"
            )

        # ---------------------------------------------------
        # DOWNLOAD BUTTON
        # ---------------------------------------------------

        st.markdown("---")

        if mode != "Remove Background":

            result_bgr = cv2.cvtColor(
                result_image.astype(np.uint8),
                cv2.COLOR_RGB2BGR
            )

            cv2.imwrite(
                "outputs/result.png",
                result_bgr
            )

        else:

            cv2.imwrite(
                "outputs/result.png",
                cv2.cvtColor(
                    result_image,
                    cv2.COLOR_RGBA2BGRA
                )
            )

        with open(
            "outputs/result.png",
            "rb"
        ) as file:

            st.download_button(
                label="⬇️ Download Result",
                data=file,
                file_name="visionai_output.png",
                mime="image/png"
            )

else:

    st.markdown("""
    <div style="
        border: 2px dashed #30363D;
        padding: 80px;
        border-radius: 20px;
        text-align: center;
        margin-top: 40px;
        background-color: #161B22;
    ">
        <h2 style="color:white;">
            Upload an Image to Start
        </h2>

        <p style="color:#8B949E;">
            AI-powered segmentation using YOLOv8 + Meta SAM
        </p>
    </div>
    """, unsafe_allow_html=True)