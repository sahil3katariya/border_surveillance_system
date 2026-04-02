import streamlit as st
import tempfile
import base64

from src.pipeline.video_pipeline import VideoPipeline
from src.pipeline.predict_pipeline import Predict_pipeline

# ------------------ CONFIG ------------------
st.set_page_config(layout="wide")

# ------------------ SESSION ------------------
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# ------------------ HEADER ------------------
st.title("🚨 AI Border Surveillance System")
st.markdown("<br>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Settings")

if not st.session_state.model_loaded:
    if st.sidebar.button("Load Model"):
        st.session_state.model_loaded = True
        st.rerun()

    st.info("👉 Please click 'Load Model' to start")
    st.stop()

mode = st.sidebar.radio(
    "Select Mode",
    ["Video Prediction", "Manual Prediction"]
)

# =========================================================
# 🎥 VIDEO MODE
# =========================================================
def show_video(video_bytes):
    video_base64 = base64.b64encode(video_bytes).decode()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:center;">
                <video autoplay muted loop controls 
                       style="width:500px;height:400px; border-radius:10px;">
                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                </video>
            </div>
            """,
            unsafe_allow_html=True
        )


def video_mode():
    st.subheader("🎥 Video Analysis")
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Video", type=["mp4"])

    if uploaded_file:

        st.success("✅ Video uploaded successfully")
        st.markdown("<br>", unsafe_allow_html=True)

        # -------- READ VIDEO --------
        video_bytes = uploaded_file.read()

        # -------- DISPLAY VIDEO --------
        show_video(video_bytes)

        st.markdown("<br>", unsafe_allow_html=True)

        # -------- SAVE TEMP FILE --------
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_bytes)

        # -------- RUN PIPELINE --------
        with st.spinner("🔍 Running YOLO Detection..."):
            pipeline = VideoPipeline()
            result = pipeline.run(tfile.name)

        # -------- OUTPUT --------
        st.subheader("📊 Detection Output")

        if result:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Object", result["object"])
                st.metric("Zone", result["zone"])

            with col2:
                st.metric("Speed", round(result["speed"], 2))
                st.metric("Time", result["time"])

            # -------- ML --------
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🧠 Risk Analysis")

            predictor = Predict_pipeline()
            risk = predictor.predict(result)

            # -------- ALERT --------
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("🚨 Final Alert")

            if risk == "High":
                st.error("🚨 HIGH RISK DETECTED!")
            elif risk == "Medium":
                st.warning("⚠️ MEDIUM RISK")
            else:
                st.success("✅ LOW RISK")

        else:
            st.info("No significant detection found")



# =========================================================
# 🧠 MANUAL MODE
# =========================================================
def manual_mode():
    st.subheader("🧠 Manual Risk Prediction")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        obj = st.selectbox("Object Type", ["Human", "Vehicle", "Animal"])
        zone = st.selectbox("Zone", ["Z1", "Z2", "Z3"])

    with col2:
        speed = st.slider("Speed", 0, 50, 10)
        time = st.selectbox("Time of Day", ["Day", "Night"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Risk"):

        data = {
            "object": obj,
            "time": time,
            "zone": zone,
            "speed": speed
        }

        predictor = Predict_pipeline()
        risk = predictor.predict(data)

        st.subheader("🚨 Prediction Result")

        if risk == "High":
            st.error("🚨 HIGH RISK DETECTED!")
        elif risk == "Medium":
            st.warning("⚠️ MEDIUM RISK")
        else:
            st.success("✅ LOW RISK")


# =========================================================
# 🚀 ROUTER
# =========================================================
if mode == "Video Prediction":
    video_mode()

elif mode == "Manual Prediction":
    manual_mode()
