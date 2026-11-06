"""Streamlit frontend for the Warehouse Box Counter.

Photos and video use separate logic:
  * Photos -> colour detector (from the fullstack project) or your YOLO model, plus room/capacity estimate
  * Video  -> YOLO model, optional tracking for a unique box count

Run:  streamlit run frontend/app.py      (backend URL from API_URL, default http://localhost:8000)
"""
import base64
import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]
IMGSZ_CHOICES = [320, 480, 640, 800, 960, 1280]
ENGINE_CHOICES = {
    "Auto (colour detector for photos)": "auto",
    "Colour detector (orange/red cartons)": "classic",
    "YOLO model": "yolo",
}
ENGINE_NAMES = {"classic": "Colour detector", "yolo": "YOLO model"}

st.set_page_config(page_title="Warehouse Box Counter", page_icon="📦", layout="wide")


# ----------------------------------------------------------------------------- API helpers
def error_text(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail", resp.text)
    except Exception:
        return resp.text
    return detail if isinstance(detail, str) else str(detail)


def backend_health():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📦 Box Counter")
    health = backend_health()
    if health is None:
        st.error(f"Cannot reach the API at {API_URL}")
    else:
        st.markdown("**Status**")
        st.write(("✅" if health["classic_available"] else "❌") + " Colour detector (photos)")
        st.write(("✅" if health["model_loaded"] else "❌") + " YOLO model (video, optional for photos)")
        if not health["model_loaded"]:
            if health.get("detail"):
                st.caption(health["detail"])
            if st.button("Reload model"):
                requests.post(f"{API_URL}/model/reload", timeout=30)
                st.rerun()

    st.subheader("Carton size (for room estimate)")
    st.caption("Box size in cm. Used for photos only.")
    b1, b2, b3 = st.columns(3)
    box_h = b1.number_input("Height", 1.0, 500.0, 25.0, 1.0)
    box_w = b2.number_input("Width", 1.0, 500.0, 30.0, 1.0)
    box_b = b3.number_input("Base", 1.0, 500.0, 35.0, 1.0)
    with st.expander("Camera (advanced)"):
        hfov = st.slider("Horizontal field of view (deg)", 40, 120, 70, 5,
                         help="Phone main camera is about 65-75. Wide CCTV lenses are 90-110. "
                              "A wrong value scales every room estimate.")
    st.caption(f"API: {API_URL}")

st.title("Warehouse Box Counter")
st.caption("Count boxes in warehouse photos and videos.")

if health is None:
    st.info("Start the backend first:  `uvicorn backend.main:app --port 8000`")
    st.stop()
if not health["classic_available"] and not health["model_loaded"]:
    st.info("No detector is available. Install scipy and scikit-image, and/or put `best.pt` in `models/`.")
    st.stop()

tab_image, tab_video, tab_about = st.tabs(["📷 Photos", "🎬 Video", "ℹ️ About"])


# ----------------------------------------------------------------------------- photos
def run_image_detection(files, engine, settings):
    results = []
    bar = st.progress(0.0, text="Detecting boxes...")
    for i, f in enumerate(files):
        try:
            resp = requests.post(
                f"{API_URL}/predict/image",
                files={"file": (f.name, f.getvalue(), f.type or "application/octet-stream")},
                data={"engine": engine, "include_image": "true",
                      "box_h": box_h, "box_w": box_w, "box_b": box_b, "hfov": hfov, **settings},
                timeout=180,
            )
            if resp.ok:
                item = resp.json()
                item["name"] = f.name
                results.append(item)
            else:
                st.error(f"{f.name}: {error_text(resp)}")
        except requests.RequestException as exc:
            st.error(f"{f.name}: request failed ({exc})")
        bar.progress((i + 1) / len(files), text=f"Processed {i + 1} of {len(files)}")
    bar.empty()
    return results


def show_room_and_capacity(r):
    st.markdown("**Estimated room (from the photo)**")
    room, cap = r.get("room"), r.get("capacity")
    if not room:
        st.info("Not enough boxes were detected to judge the scale of the photo, so the room size "
                "cannot be estimated.")
        return
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Length (depth)", f"{room['length_m']:.1f} m")
    e2.metric("Width", f"{room['width_m']:.1f} m")
    e3.metric("Stack height", f"{room['height_m']:.2f} m")
    e4.metric("Camera to stacks", f"{room['camera_distance_m']:.1f} m")
    st.caption(f"Based on {room['boxes_used']} boxes, {room['layers_visible']} visible layers, "
               f"confidence: {room['confidence']}. " + " ".join(room["notes"]))
    if cap:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Room floor area", f"{cap['floor_area_m2']:,.0f} m²")
        c2.metric("Boxes per layer", f"{cap['boxes_per_layer']:,}")
        c3.metric("Layers", cap["layers"])
        c4.metric("Expected capacity (boxes)", f"{cap['capacity']:,}")
        st.progress(min(cap["fill_pct"] / 100, 1.0))
        st.write(f"Detected **{cap['detected']:,}** of about **{cap['capacity']:,}** boxes the room can hold "
                 f"(**{cap['fill_pct']:.1f}%** full, room for about **{cap['remaining']:,}** more).")
        st.caption("Capacity = floor grid of carton footprints x 60% usable floor x stack layers. "
                   "Detected boxes are only the ones visible, so real stock may be higher.")


with tab_image:
    engine_label = st.selectbox("Photo engine", list(ENGINE_CHOICES), index=0,
                                help="Auto uses the colour detector. Choose YOLO to use your trained model.")
    engine = ENGINE_CHOICES[engine_label]
    effective = "yolo" if engine == "yolo" else "classic"
    if effective == "yolo" and not health["model_loaded"]:
        st.warning("YOLO model is not loaded. Put `best.pt` in `models/` and press Reload model.")
    if effective == "classic" and not health["classic_available"]:
        st.warning("The colour detector is not available (scipy / scikit-image missing).")

    with st.expander("Photo settings", expanded=False):
        if effective == "classic":
            a_thr = st.slider("Carton colour threshold", 125, 150, 134)
            seam = st.slider("Seam sensitivity (lower = more splits)", 4, 25, 12)
            min_dist = st.slider("Min distance between boxes", 3, 15, 7)
            photo_settings = {"a_thr": a_thr, "seam": seam, "min_dist": min_dist}
        else:
            p_conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="p_conf")
            p_iou = st.slider("Overlap filter (IoU)", 0.10, 0.90, 0.50, 0.05, key="p_iou")
            p_size = st.select_slider("Inference size", options=IMGSZ_CHOICES, value=960, key="p_size")
            photo_settings = {"conf": p_conf, "iou": p_iou, "imgsz": p_size}

    if effective == "classic":
        st.caption("The colour detector only recognises orange/red cartons. Plain brown cardboard "
                   "and very dense stacks are missed, so treat the number as approximate. "
                   "Switch the engine to YOLO for those.")

    files = st.file_uploader("Upload one or more photos", type=IMAGE_TYPES,
                             accept_multiple_files=True, key="image_uploader")
    if files and st.button("Detect boxes", type="primary", key="detect_images"):
        st.session_state["image_results"] = run_image_detection(files, engine, photo_settings)

    results = st.session_state.get("image_results")
    if results:
        summary = pd.DataFrame([
            {"image": r["name"], "engine": ENGINE_NAMES.get(r["engine"], r["engine"]),
             "boxes": r["count"], "inference_ms": r["inference_ms"]} for r in results
        ])
        c1, c2, c3 = st.columns(3)
        c1.metric("Images", len(results))
        c2.metric("Total boxes", int(summary["boxes"].sum()))
        c3.metric("Avg time (ms)", round(float(summary["inference_ms"].mean()), 1))
        st.dataframe(summary)
        st.download_button("Download counts (CSV)", summary.to_csv(index=False).encode("utf-8"),
                           file_name="box_counts.csv", mime="text/csv")

        for r in results:
            st.divider()
            st.subheader(f"{r['name']}: {r['count']} boxes")
            st.caption(f"Engine: {ENGINE_NAMES.get(r['engine'], r['engine'])}")
            if r.get("annotated_image_b64"):
                st.image(base64.b64decode(r["annotated_image_b64"]))
            show_room_and_capacity(r)
            with st.expander("Detection details"):
                if r["detections"]:
                    st.dataframe(pd.DataFrame(r["detections"]).drop(columns=["track_id"], errors="ignore"))
                else:
                    st.write("No boxes detected.")


# ----------------------------------------------------------------------------- video
def run_video_job(video, settings):
    resp = requests.post(
        f"{API_URL}/predict/video",
        files={"file": (video.name, video.getvalue(), video.type or "application/octet-stream")},
        data=settings,
        timeout=600,
    )
    if not resp.ok:
        st.error(error_text(resp))
        return None
    job_id = resp.json()["job_id"]

    bar = st.progress(0.0, text="Queued...")
    while True:
        status = requests.get(f"{API_URL}/jobs/{job_id}", timeout=15).json()
        bar.progress(min(float(status["progress"]), 1.0), text=status["message"])
        if status["status"] == "done":
            break
        if status["status"] == "failed":
            bar.empty()
            st.error(f"Processing failed: {status.get('error')}")
            return None
        time.sleep(1)
    bar.empty()

    video_bytes = requests.get(f"{API_URL}{status['result']['video_url']}", timeout=300).content
    csv_bytes = requests.get(f"{API_URL}{status['result']['csv_url']}", timeout=30).content
    return {"summary": status["result"], "video": video_bytes, "csv": csv_bytes, "name": video.name}


with tab_video:
    if not health["model_loaded"]:
        st.warning("Video uses the YOLO model. Put `best.pt` in `models/` and press Reload model in the sidebar.")
    else:
        st.caption("Video always uses the YOLO model.")
        with st.expander("Video settings", expanded=False):
            v_conf = st.slider("Confidence threshold", 0.05, 0.95, 0.50, 0.05, key="v_conf")
            v_iou = st.slider("Overlap filter (IoU)", 0.10, 0.90, 0.40, 0.05, key="v_iou")
            v_size = st.select_slider("Inference size", options=IMGSZ_CHOICES, value=640, key="v_size")

        video = st.file_uploader("Upload a video", type=VIDEO_TYPES, key="video_uploader")
        o1, o2 = st.columns(2)
        track = o1.checkbox("Count unique boxes (tracking)", value=True,
                            help="Gives each box an ID so a box moving through the frame is counted once. "
                                 "Best for conveyor belts. Turn off to get the number of boxes visible per frame.")
        frame_stride = o2.slider("Analyze every Nth frame", 1, 10, 1,
                                 help="Higher = faster, but less precise. Keep at 1-2 when tracking.")
        if video:
            st.video(video)
        if video and st.button("Analyze video", type="primary", key="analyze_video"):
            st.session_state["video_result"] = run_video_job(video, {
                "conf": v_conf, "iou": v_iou, "imgsz": v_size,
                "frame_stride": frame_stride, "track": "true" if track else "false",
            })

        vr = st.session_state.get("video_result")
        if vr:
            s = vr["summary"]
            st.subheader(f"Results for {vr['name']}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Frames analyzed", s["frames_processed"])
            m2.metric("Max boxes in a frame", s["max_count"])
            m3.metric("Average per frame", s["avg_count"])
            if s["unique_count"] is not None:
                m4.metric("Total boxes (unique)", s["unique_count"])
            else:
                m4.metric("Total boxes", "Turn on tracking")

            if s["browser_playable"]:
                st.video(vr["video"], format="video/mp4")
            else:
                st.warning("The annotated video could not be re-encoded for the browser. Download it to watch it.")
            st.download_button("Download annotated video", vr["video"], file_name="annotated.mp4", mime="video/mp4")

            frames_df = pd.DataFrame(s["per_frame"])
            st.markdown("**Boxes per frame**")
            st.line_chart(frames_df, x="time_sec", y="count")
            st.download_button("Download per-frame counts (CSV)", vr["csv"],
                               file_name="box_counts.csv", mime="text/csv")


# ----------------------------------------------------------------------------- about
with tab_about:
    st.markdown(
        """
**How it works**

- **Photos** are analysed by the *colour detector* (no training needed, works on orange/red cartons)
  or by your *YOLO model*. The app also estimates the room size and capacity from the carton size.
- **Video** is analysed by the YOLO model as a background job, with optional tracking for a unique count.
- The two paths are separate, so changing one never affects the other.

**Getting accurate counts**

- The colour detector cannot see plain brown cartons. For those, train a YOLO model on labeled photos
  from your own cameras and select *YOLO model* as the photo engine.
- With YOLO, a larger *Inference size* finds small or distant boxes. Lower the confidence if boxes are missed.
- Counts include only **visible** boxes. Boxes hidden behind others cannot be counted.
- Room size and capacity are rough estimates (about +/-30%).
        """
    )
