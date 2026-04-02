import streamlit as st
import cv2
import tempfile
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Border Surveillance AI",
    layout="wide"
)

# ------------------ STYLING ------------------
st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: bold;
    color: #FF4B4B;
}
.alert-red {
    background-color: #FF4B4B;
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}
.alert-green {
    background-color: #28a745;
    color: white;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}
.card {
    background-color: #161A23;
    padding: 12px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("🚨 Surveillance AI")

mode = st.sidebar.radio(
    "Select Mode",
    ["🎥 Predict with Video", "🖼 Predict with Image"]
)

st.sidebar.markdown("---")
st.sidebar.info("AI Border Surveillance System")


# ------------------ HEADER ------------------
st.markdown('<div class="main-title">🚨 Border Surveillance Dashboard</div>', unsafe_allow_html=True)
st.markdown("Real-time detection with zones + alerts")

st.markdown("---")


# ================= VIDEO MODE =================
if mode == "🎥 Predict with Video":

    col_ctrl1, col_ctrl2 = st.columns([3,1])

    with col_ctrl1:
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

    with col_ctrl2:
        start = st.button("▶ Start")

    st.markdown("---")

    col1, col2 = st.columns([3,1])

    with col1:
        st.subheader("📹 Processed Feed")
        frame_window = st.empty()

    with col2:
        st.subheader("🚨 Alert")
        alert_box = st.empty()

        st.subheader("📊 Stats")
        stats_box = st.empty()

    # -------- PROCESS VIDEO --------
    if video_file and start:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)

        human, vehicle, animal = 0, 0, 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 🔥 ============ YOUR YOLO PROCESSING HERE ============
            # Replace below with your actual detection code

            height, width = frame.shape[:2]

            # Fake zone lines
            line1 = int(height * 0.4)
            line2 = int(height * 0.75)

            cv2.line(frame, (0, line1), (width, line1), (255,255,0), 2)
            cv2.line(frame, (0, line2), (width, line2), (255,0,0), 2)

            # Fake detection example
            cv2.rectangle(frame, (100,100), (300,300), (0,255,0), 2)
            cv2.putText(frame, "Human | Z2 | 10 km/h", (100,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Fake condition
            danger = False  # 🔥 replace with your Z3 logic

            # ====================================================

            # Show frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame, channels="RGB")

            # Alert UI
            if danger:
                alert_box.markdown(
                    '<div class="alert-red">🚨 DANGER DETECTED (Z3)</div>',
                    unsafe_allow_html=True
                )
            else:
                alert_box.markdown(
                    '<div class="alert-green">✅ Safe</div>',
                    unsafe_allow_html=True
                )

            # Stats
            stats_box.write({
                "Humans": human,
                "Vehicles": vehicle,
                "Animals": animal
            })

            time.sleep(0.03)

        cap.release()

    else:
        frame_window.info("Upload video and click Start")


# ================= IMAGE MODE =================
elif mode == "🖼 Predict with Image":

    image_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if image_file:

        file_bytes = bytearray(image_file.read())
        frame = cv2.imdecode(
            cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
        )

        # 🔥 ============ YOUR YOLO IMAGE PROCESSING ============
        # Add detection here

        height, width = frame.shape[:2]

        line1 = int(height * 0.4)
        line2 = int(height * 0.75)

        cv2.line(frame, (0, line1), (width, line1), (255,255,0), 2)
        cv2.line(frame, (0, line2), (width, line2), (255,0,0), 2)

        # ====================================================

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Processed Image")

        st.success("Processing complete")

    else:
        st.info("Upload an image to start")