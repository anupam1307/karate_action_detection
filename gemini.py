import cv2
from ultralytics import YOLO
import numpy as np
import os

# --- Configuration & Parameters ---

# Model Selection
MODEL_PATH = "yolo11s-pose.pt" 

# Video Paths
INPUT_VIDEO_PATH = "Input_Video/input2.mp4"
OUTPUT_VIDEO_PATH = "Output_Video/output_actions_yolo11_dynamic.mp4" 

# Fighter Identification & Tracking
CONFIDENCE_THRESHOLD = 0.5 # Minimum detection confidence for YOLO
PERSON_CLASS_ID = 0        # Assuming 'person' is class 0 in your model

# Ring Boundaries (relative to frame dimensions)
RING_LEFT_RATIO = 0.1
RING_RIGHT_RATIO = 0.9
RING_TOP_RATIO = 0.1
RING_BOTTOM_RATIO = 0.9

# Filtering
MIN_BOX_HEIGHT_RATIO = 0.15 # Ignore boxes smaller than 15% of frame height

# Action Detection Thresholds (Normalized Movement - **NEEDS TUNING**)
PUNCH_MOVEMENT_THRESHOLD = 0.04
KICK_MOVEMENT_THRESHOLD = 0.06
MIN_KEYPOINT_CONFIDENCE = 0.3 # Min confidence for keypoint usage

# Colors (BGR format for OpenCV)
FIGHTER1_PUNCH_COLOR = (0, 0, 255)    # Red
FIGHTER2_PUNCH_COLOR = (255, 0, 0)    # Blue
FIGHTER1_KICK_COLOR = (203, 192, 255)  # Pink (BGR)
FIGHTER2_KICK_COLOR = (0, 165, 255)    # Orange
DEFAULT_BOX_COLOR = (0, 255, 255)      # Yellow (BGR)

# Keypoint Indices (COCO format - assumed standard)
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_ANKLE, R_ANKLE = 15, 16
NUM_KEYPOINTS = 17 # Expected number of keypoints

# --- Initialization ---

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    exit()

# Load YOLO model
try:
    model = YOLO(MODEL_PATH)
    print(f"Successfully loaded model: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLO model from '{MODEL_PATH}': {e}")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error opening video file: {INPUT_VIDEO_PATH}")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Absolute ring boundaries
ring_left = int(frame_width * RING_LEFT_RATIO)
ring_right = int(frame_width * RING_RIGHT_RATIO)
ring_top = int(frame_height * RING_TOP_RATIO)
ring_bottom = int(frame_height * RING_BOTTOM_RATIO)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
print(f"Processing video: {INPUT_VIDEO_PATH}")
print(f"Output video will be saved to: {OUTPUT_VIDEO_PATH}")

# State dictionary to store previous keypoints for *all* tracked objects
prev_keypoints = {} # {track_id: {'xy': kps_norm, 'conf': kps_conf}}

frame_count = 0
print("Processing started...")

# --- Main Processing Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    processed_frame = frame.copy() # Work on a copy

    # --- YOLO Detection with Tracking ---
    # Track only persons (class 0) for efficiency
    results = model.track(processed_frame, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD, persist=True, verbose=False)

    # Store potential fighter candidates for this frame
    fighter_candidates = []
    current_frame_keypoints = {} # Store keypoints of *all* detected persons this frame

    # --- Process Detections ---
    if (results and results[0].boxes is not None and results[0].keypoints is not None and
        hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None):

        boxes = results[0].boxes
        keypoints_data = results[0].keypoints
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for i, track_id in enumerate(track_ids):
            # --- Get Data ---
            if i >= len(keypoints_data.xyn) or i >= len(boxes.xyxy): continue

            keypoints_norm = keypoints_data.xyn[i].cpu().numpy()
            keypoints_conf = keypoints_data.conf[i].cpu().numpy() if keypoints_data.conf is not None else np.ones(keypoints_norm.shape[0])

            if keypoints_norm.shape[0] != NUM_KEYPOINTS or keypoints_conf.shape[0] != NUM_KEYPOINTS: continue

            x1, y1, x2, y2 = map(int, boxes.xyxy[i])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            box_area = (x2 - x1) * (y2 - y1)
            box_height = y2 - y1

            # --- Store keypoints for *all* tracked persons for potential future use ---
            current_frame_keypoints[track_id] = {'xy': keypoints_norm, 'conf': keypoints_conf}

            # --- Filter Candidates ---
            # 1. Within Ring?
            if not (ring_left < cx < ring_right and ring_top < cy < ring_bottom):
                continue
            # 2. Minimum Size?
            if (box_height / frame_height) < MIN_BOX_HEIGHT_RATIO:
                continue

            # Add valid candidate to list for sorting
            fighter_candidates.append({
                'track_id': track_id,
                'box': (x1, y1, x2, y2),
                'center_x': cx,
                'area': box_area,
                'keypoints_norm': keypoints_norm,
                'keypoints_conf': keypoints_conf
            })

    # --- Identify Top Two Fighters Dynamically ---
    fighter1_data = None
    fighter2_data = None

    if len(fighter_candidates) >= 2:
        # Sort candidates by area (largest first)
        fighter_candidates.sort(key=lambda x: x['area'], reverse=True)
        # Take the top two
        candidate1 = fighter_candidates[0]
        candidate2 = fighter_candidates[1]
        # Assign based on horizontal position
        if candidate1['center_x'] < candidate2['center_x']:
            fighter1_data = candidate1
            fighter2_data = candidate2
        else:
            fighter1_data = candidate2
            fighter2_data = candidate1
    elif len(fighter_candidates) == 1:
        # If only one candidate, assign as Fighter 1 for drawing purposes
        fighter1_data = fighter_candidates[0]

    # --- Process Identified Fighters ---
    fighters_to_process = []
    if fighter1_data: fighters_to_process.append((fighter1_data, "Fighter 1"))
    if fighter2_data: fighters_to_process.append((fighter2_data, "Fighter 2"))

    for fighter_data, fighter_label in fighters_to_process:
        track_id = fighter_data['track_id']
        x1, y1, x2, y2 = fighter_data['box']
        keypoints_norm = fighter_data['keypoints_norm']
        keypoints_conf = fighter_data['keypoints_conf']

        action_text = ""
        action_color = DEFAULT_BOX_COLOR
        highlight_box = False

        # --- Action Detection ---
        if track_id in prev_keypoints:
            prev_kps_data = prev_keypoints[track_id]
            prev_kps_norm = prev_kps_data['xy']
            prev_kps_conf = prev_kps_data['conf']

            if prev_kps_norm.shape[0] == NUM_KEYPOINTS and prev_kps_conf.shape[0] == NUM_KEYPOINTS: # Check prev data validity

                # 1. Punch Detection
                punch_kps_indices = [L_WRIST, R_WRIST, L_SHOULDER, R_SHOULDER]
                can_detect_punch = all(
                    keypoints_conf[kp_idx] > MIN_KEYPOINT_CONFIDENCE and
                    prev_kps_conf[kp_idx] > MIN_KEYPOINT_CONFIDENCE
                    for kp_idx in punch_kps_indices
                )
                if can_detect_punch:
                    left_wrist_move = np.linalg.norm(keypoints_norm[L_WRIST] - prev_kps_norm[L_WRIST])
                    right_wrist_move = np.linalg.norm(keypoints_norm[R_WRIST] - prev_kps_norm[R_WRIST])
                    wrist_movement = max(left_wrist_move, right_wrist_move)
                    shoulder_y_avg = (keypoints_norm[L_SHOULDER][1] + keypoints_norm[R_SHOULDER][1]) / 2
                    upper_body_line_norm = shoulder_y_avg + 0.05
                    wrist_y_norm = min(keypoints_norm[L_WRIST][1], keypoints_norm[R_WRIST][1])

                    if wrist_movement > PUNCH_MOVEMENT_THRESHOLD and wrist_y_norm < upper_body_line_norm:
                        action_text = "PUNCH"
                        action_color = FIGHTER1_PUNCH_COLOR if fighter_label == "Fighter 1" else FIGHTER2_PUNCH_COLOR
                        highlight_box = True

                # 2. Kick Detection
                kick_kps_indices = [L_ANKLE, R_ANKLE]
                can_detect_kick = all(
                    keypoints_conf[kp_idx] > MIN_KEYPOINT_CONFIDENCE and
                    prev_kps_conf[kp_idx] > MIN_KEYPOINT_CONFIDENCE
                    for kp_idx in kick_kps_indices
                )
                if can_detect_kick:
                    left_ankle_move = np.linalg.norm(keypoints_norm[L_ANKLE] - prev_kps_norm[L_ANKLE])
                    right_ankle_move = np.linalg.norm(keypoints_norm[R_ANKLE] - prev_kps_norm[R_ANKLE])
                    ankle_movement = max(left_ankle_move, right_ankle_move)

                    if ankle_movement > KICK_MOVEMENT_THRESHOLD:
                        action_text = "KICK" # Kick overrides punch label/color
                        action_color = FIGHTER1_KICK_COLOR if fighter_label == "Fighter 1" else FIGHTER2_KICK_COLOR
                        highlight_box = True

        # --- Drawing ---
        box_thickness = 3 if highlight_box else 2
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), action_color, box_thickness)

        label_pos = (x1, y1 - 10 if y1 > 30 else y1 + 20)
        cv2.putText(processed_frame, fighter_label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, DEFAULT_BOX_COLOR, 2, cv2.LINE_AA)

        if action_text:
            action_pos = (label_pos[0], label_pos[1] + 20)
            cv2.putText(processed_frame, action_text, action_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2, cv2.LINE_AA)


    # --- Update State for Next Frame ---
    # Store keypoints of *all* currently detected persons, not just the top two
    prev_keypoints = current_frame_keypoints.copy()

    # Write the frame
    out.write(processed_frame)

    # Optional: Display
    # cv2.imshow("Fighter Action Detection", processed_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

# --- Cleanup ---
cap.release()
out.release()
# cv2.destroyAllWindows()

print("-" * 30)
print(f"Processing complete.")
print(f"Output video saved as: {OUTPUT_VIDEO_PATH}")
print(f"Model used: {MODEL_PATH}")
print("** Dynamic fighter identification implemented. Remember to tune thresholds! **")
print("-" * 30)