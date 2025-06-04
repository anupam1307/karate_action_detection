import cv2
from ultralytics import YOLO
import numpy as np
import os
from datetime import datetime

# --- Configuration & Parameters ---
MODEL_PATH = "yolo11s-pose.pt"
INPUT_VIDEO_PATH = "Input_Video/input.mp4"
OUTPUT_VIDEO_PATH = f"Output_Video/output_actions_yolo11_refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

# Detection and Tracking
CONFIDENCE_THRESHOLD = 0.5  # Lowered for better detection
PERSON_CLASS_ID = 0
IOU_THRESHOLD = 0.7  # Slightly lowered for tracking stability
MIN_BOX_HEIGHT_RATIO = 0.2  # Relaxed to capture fighters
RING_LEFT_RATIO, RING_RIGHT_RATIO = 0.1, 0.9  # Widened ring area
RING_TOP_RATIO, RING_BOTTOM_RATIO = 0.1, 0.9

# Color Detection for Pajamas (HSV ranges, adjusted for lighting)
WHITE_PAJAMA_HSV = ([0, 0, 180], [180, 40, 255])  # Wider range for white
BLACK_PAJAMA_HSV = ([0, 0, 0], [180, 255, 60])    # Wider range for black
COLOR_CONFIDENCE = 0.4  # Lowered for better detection

# Action Detection
PUNCH_MOVEMENT_THRESHOLD = 0.03
KICK_MOVEMENT_THRESHOLD = 0.05
MIN_KEYPOINT_CONFIDENCE = 0.4  # Slightly lowered
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_ANKLE, R_ANKLE = 15, 16
NUM_KEYPOINTS = 17

# Visualization
FIGHTER1_PUNCH_COLOR = (0, 0, 255)    # Red
FIGHTER2_PUNCH_COLOR = (255, 0, 0)    # Blue
FIGHTER1_KICK_COLOR = (203, 192, 255) # Pink
FIGHTER2_KICK_COLOR = (0, 165, 255)   # Orange
DEFAULT_BOX_COLOR = (0, 255, 255)     # Yellow
TEXT_BOX_COLOR = (0, 0, 0, 200)       # Semi-transparent black
TEXT_COLOR = (255, 255, 255)          # White text

# --- Helper Functions ---
def is_color_match(image, box, hsv_lower, hsv_upper):
    try:
        x1, y1, x2, y2 = map(int, box)
        y1 = max(y1 + int((y2 - y1) * 0.6), y1)  # Focus on lower 40% (pajamas)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return False
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array(hsv_lower), np.array(hsv_upper))
        color_pixels = cv2.countNonZero(mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        return color_pixels / total_pixels > COLOR_CONFIDENCE
    except Exception as e:
        print(f"Error in color detection: {e}")
        return False

# --- Initialization ---
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'")
    exit()

try:
    model = YOLO(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"Error opening video: {INPUT_VIDEO_PATH}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ring_left = int(frame_width * RING_LEFT_RATIO)
ring_right = int(frame_width * RING_RIGHT_RATIO)
ring_top = int(frame_height * RING_TOP_RATIO)
ring_bottom = int(frame_height * RING_BOTTOM_RATIO)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
print(f"Processing video: {INPUT_VIDEO_PATH}")
print(f"Output saved to: {OUTPUT_VIDEO_PATH}")

prev_keypoints = {}
frame_count = 0
print("Processing started...")

# --- Main Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_count += 1
    processed_frame = frame.copy()

    # Debug specific frame (around 00:22, assuming 30 FPS)
    if frame_count in range(650, 670):
        cv2.imwrite(f"debug_frame_{frame_count}.jpg", processed_frame)

    # YOLO Detection with Tracking
    results = model.track(processed_frame, classes=[PERSON_CLASS_ID], conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD, persist=True, verbose=False)

    fighter_candidates = []
    current_frame_keypoints = {}

    if (results and results[0].boxes is not None and results[0].keypoints is not None and
        hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None):

        boxes = results[0].boxes.xyxy.cpu().numpy()
        keypoints_data = results[0].keypoints
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for i, track_id in enumerate(track_ids):
            if i >= len(keypoints_data.xyn) or i >= len(boxes):
                continue

            keypoints_norm = keypoints_data.xyn[i].cpu().numpy()
            keypoints_conf = keypoints_data.conf[i].cpu().numpy() if keypoints_data.conf is not None else np.ones(NUM_KEYPOINTS)

            if keypoints_norm.shape[0] != NUM_KEYPOINTS or keypoints_conf.shape[0] != NUM_KEYPOINTS:
                continue

            x1, y1, x2, y2 = boxes[i]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            box_height = y2 - y1

            # Relaxed Filtering
            if not (ring_left < cx < ring_right and ring_top < cy < ring_bottom) or (box_height / frame_height) < MIN_BOX_HEIGHT_RATIO:
                continue

            # Color Detection
            is_white = is_color_match(frame, (x1, y1, x2, y2), WHITE_PAJAMA_HSV[0], WHITE_PAJAMA_HSV[1])
            is_black = is_color_match(frame, (x1, y1, x2, y2), BLACK_PAJAMA_HSV[0], BLACK_PAJAMA_HSV[1])

            # Debug specific frame
            if frame_count in range(650, 670):
                print(f"Frame {frame_count}: Track ID {track_id}, White: {is_white}, Black: {is_black}, Box: ({x1}, {y1}, {x2}, {y2})")

            if is_white or is_black:
                fighter_candidates.append({
                    'track_id': track_id,
                    'box': (x1, y1, x2, y2),
                    'center_x': cx,
                    'is_white': is_white,
                    'is_black': is_black,
                    'keypoints_norm': keypoints_norm,
                    'keypoints_conf': keypoints_conf
                })
                current_frame_keypoints[track_id] = {'xy': keypoints_norm, 'conf': keypoints_conf}

    # Assign Fighters with Fallback
    fighter1_data = None  # White pajama
    fighter2_data = None  # Black pajama

    # First pass: strict assignment
    for candidate in fighter_candidates:
        try:
            if candidate['is_white'] and not fighter1_data and not candidate['is_black']:
                fighter1_data = candidate
            elif candidate['is_black'] and not fighter2_data and not candidate['is_white']:
                fighter2_data = candidate
        except KeyError as e:
            print(f"Frame {frame_count}: KeyError in fighter assignment: {e}, candidate: {candidate}")
            continue

    # Fallback: if no fighters detected, relax color constraint and pick largest boxes
    if not fighter1_data and not fighter2_data and fighter_candidates:
        fighter_candidates.sort(key=lambda x: (x['box'][2] - x['box'][0]) * (x['box'][3] - x['box'][1]), reverse=True)
        for candidate in fighter_candidates[:2]:  # Take top 2 by area
            if candidate['center_x'] < frame_width / 2 and not fighter1_data:
                fighter1_data = candidate
            elif candidate['center_x'] >= frame_width / 2 and not fighter2_data:
                fighter2_data = candidate

    # Process Fighters
    fighters = []
    if fighter1_data:
        fighters.append((fighter1_data, "Fighter 1", FIGHTER1_PUNCH_COLOR, FIGHTER1_KICK_COLOR))
    if fighter2_data:
        fighters.append((fighter2_data, "Fighter 2", FIGHTER2_PUNCH_COLOR, FIGHTER2_KICK_COLOR))

    action_texts = []
    for fighter_data, label, punch_color, kick_color in fighters:
        track_id = fighter_data['track_id']
        x1, y1, x2, y2 = map(int, fighter_data['box'])
        keypoints_norm = fighter_data['keypoints_norm']
        keypoints_conf = fighter_data['keypoints_conf']

        action_text = ""
        action_color = DEFAULT_BOX_COLOR
        highlight_box = False

        if track_id in prev_keypoints:
            prev_kps = prev_keypoints[track_id]['xy']
            prev_conf = prev_keypoints[track_id]['conf']

            if prev_kps.shape[0] == NUM_KEYPOINTS and prev_conf.shape[0] == NUM_KEYPOINTS:
                # Punch Detection
                punch_kps = [L_WRIST, R_WRIST, L_SHOULDER, R_SHOULDER]
                can_detect_punch = all(keypoints_conf[i] > MIN_KEYPOINT_CONFIDENCE and prev_conf[i] > MIN_KEYPOINT_CONFIDENCE for i in punch_kps)
                if can_detect_punch:
                    wrist_move = max(
                        np.linalg.norm(keypoints_norm[L_WRIST] - prev_kps[L_WRIST]),
                        np.linalg.norm(keypoints_norm[R_WRIST] - prev_kps[R_WRIST])
                    )
                    shoulder_y = (keypoints_norm[L_SHOULDER][1] + keypoints_norm[R_SHOULDER][1]) / 2
                    wrist_y = min(keypoints_norm[L_WRIST][1], keypoints_norm[R_WRIST][1])
                    if wrist_move > PUNCH_MOVEMENT_THRESHOLD and wrist_y < shoulder_y + 0.05:
                        action_text = "PUNCH"
                        action_color = punch_color
                        highlight_box = True

                # Kick Detection
                kick_kps = [L_ANKLE, R_ANKLE]
                can_detect_kick = all(keypoints_conf[i] > MIN_KEYPOINT_CONFIDENCE and prev_conf[i] > MIN_KEYPOINT_CONFIDENCE for i in kick_kps)
                if can_detect_kick:
                    ankle_move = max(
                        np.linalg.norm(keypoints_norm[L_ANKLE] - prev_kps[L_ANKLE]),
                        np.linalg.norm(keypoints_norm[R_ANKLE] - prev_kps[R_ANKLE])
                    )
                    if ankle_move > KICK_MOVEMENT_THRESHOLD:
                        action_text = "KICK"
                        action_color = kick_color
                        highlight_box = True

        # Draw Bounding Box
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), action_color, 3 if highlight_box else 2)
        cv2.putText(processed_frame, label, (x1, y1 - 10 if y1 > 30 else y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, DEFAULT_BOX_COLOR, 2, cv2.LINE_AA)

        if action_text:
            action_texts.append(f"{label}: {action_text}")

    # Draw Action Text Box
    if action_texts:
        text_box_y = frame_height - 100
        text_box_x = 20
        text_box_width = 300
        text_box_height = 60
        overlay = processed_frame.copy()
        cv2.rectangle(overlay, (text_box_x, text_box_y), (text_box_x + text_box_width, text_box_y + text_box_height), TEXT_BOX_COLOR, -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0, processed_frame)
        for i, text in enumerate(action_texts):
            cv2.putText(processed_frame, text, (text_box_x + 10, text_box_y + 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

    prev_keypoints = current_frame_keypoints.copy()
    out.write(processed_frame)

    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

# --- Cleanup ---
cap.release()
out.release()
print(f"Processing complete. Output saved to: {OUTPUT_VIDEO_PATH}")