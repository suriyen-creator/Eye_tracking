import cv2
import numpy as np
import time
import csv
import os
import mediapipe as mp
import random

# ---------------------- ปรับแต่งได้ ----------------------
CAM_INDEX = 0
FLIP = True
BOX_W, BOX_H = 100, 100          # กล่องที่ “ตามองไป” (สำหรับแสดงและจับภายหลังคาลิเบรต)
TARGET_W, TARGET_H = 160, 160     # กรอบ "เป้าหมาย" สำหรับเทสต์
SMOOTH_ALPHA = 0.25
MIN_IRIS_AREA = 6.0
BTN_TEXT = "ยืนยัน"
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Gaze Focus Test (9pt Calib + Two Eyes)"
START_FULLSCREEN = True
PAD_RATIO = 0.08                   # เว้นขอบตอนวางจุดคาลิเบรต
# การทดสอบ
REQUIRED_CONTINUOUS_SEC = 5.0      # ต้องมองต่อเนื่องกี่วินาที ถึงจะนับว่าผ่านใน 1 รอบ
N_TRIALS = 5                       # จำนวนรอบทดสอบ
RESULTS_CSV = "focus_results.csv"  # ไฟล์ผลลัพธ์
# ---------------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS  = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

def iris_center_and_area(landmarks, idx_list, w, h):
    try:
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idx_list], dtype=np.float32)
        cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())
        x, y = pts[:,0], pts[:,1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        return cx, cy, float(area)
    except Exception:
        return None, None, 0.0

def clamp_box(x, y, w, h, bw, bh):
    left = int(x - bw / 2); top = int(y - bh / 2)
    left = max(0, min(left, w - bw)); top = max(0, min(top, h - bh))
    return left, top, left + bw, top + bh

def draw_target_dot(frame, pt, label):
    cv2.circle(frame, pt, 10, (0, 0, 255), -1)
    cv2.circle(frame, pt, 25, (0, 0, 255), 2)
    cv2.putText(frame, label, (pt[0] + 15, pt[1] - 15), FONT, 0.8, (0,0,255), 2)

def draw_button(frame, rect, text=BTN_TEXT):
    (x1,y1,x2,y2) = rect
    cv2.rectangle(frame, (x1,y1), (x2,y2), (60,60,60), -1)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.8, 2)
    tx = x1 + (x2 - x1 - tw)//2
    ty = y1 + (y2 - y1 + th)//2
    cv2.putText(frame, text, (tx, ty), FONT, 0.8, (255,255,255), 2)

def perspective_map(H, pt):
    src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, H)
    return int(dst[0,0,0]), int(dst[0,0,1])

def set_fullscreen(full=True):
    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL
    )

def get_combined_gaze(landmarks, fw, fh):
    lx, ly, la = iris_center_and_area(landmarks, LEFT_IRIS,  fw, fh)
    rx, ry, ra = iris_center_and_area(landmarks, RIGHT_IRIS, fw, fh)
    left_ok  = (lx is not None and la >= MIN_IRIS_AREA)
    right_ok = (rx is not None and ra >= MIN_IRIS_AREA)
    if left_ok and right_ok:
        return ( (lx+rx)/2.0, (ly+ry)/2.0, "both" )
    elif right_ok:
        return ( rx, ry, "right" )
    elif left_ok:
        return ( lx, ly, "left" )
    else:
        return ( None, None, "none" )

def rand_target_rect(fw, fh, tw, th, margin=40):
    # สุ่มกรอบเป้าหมายให้อยู่ในจอ
    x1 = random.randint(margin, max(margin, fw - tw - margin))
    y1 = random.randint(margin, max(margin, fh - th - margin))
    return (x1, y1, x1 + tw, y1 + th)

def inside_rect(x, y, rect):
    x1,y1,x2,y2 = rect
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def save_results_csv(rows, path=RESULTS_CSV):
    header = ["trial", "passed", "time_to_pass_sec"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("ไม่พบกล้อง/เปิดกล้องไม่ได้")

    prev_gx, prev_gy = None, None
    H = None
    confirmed_click = False
    fullscreen = START_FULLSCREEN

    def on_mouse(event, x, y, flags, userdata):
        nonlocal confirmed_click
        btn_rect = userdata["btn_rect"]
        if event == cv2.EVENT_LBUTTONDOWN:
            x1,y1,x2,y2 = btn_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                confirmed_click = True

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        set_fullscreen(fullscreen)
        userdata = {"btn_rect": (0,0,0,0)}
        cv2.setMouseCallback(WINDOW_NAME, on_mouse, userdata)

        ok, frame = cap.read()
        if not ok:
            cap.release(); cv2.destroyAllWindows(); return
        if FLIP: frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # --------- 9 จุดคาลิเบรต ---------
        pad_x, pad_y = int(PAD_RATIO*w), int(PAD_RATIO*h)
        cx, cy = w//2, h//2
        targets = [
            (cx, cy, "Center"),
            (pad_x, pad_y, "Top-Left"),
            (w - pad_x, pad_y, "Top-Right"),
            (pad_x, h - pad_y, "Bottom-Left"),
            (w - pad_x, h - pad_y, "Bottom-Right"),
            (cx, pad_y, "Top-Center"),
            (cx, h - pad_y, "Bottom-Center"),
            (pad_x, cy, "Left-Center"),
            (w - pad_x, cy, "Right-Center"),
        ]

        target_points, gaze_points = [], []

        idx = 0
        while idx < len(targets):
            ok, frame = cap.read()
            if not ok: break
            if FLIP: frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            gaze = None
            used = "none"
            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0].landmark
                gx, gy, used = get_combined_gaze(face, fw, fh)
                if gx is not None:
                    if prev_gx is None:
                        prev_gx, prev_gy = gx, gy
                    else:
                        prev_gx = SMOOTH_ALPHA * gx + (1 - SMOOTH_ALPHA) * prev_gx
                        prev_gy = SMOOTH_ALPHA * gy + (1 - SMOOTH_ALPHA) * prev_gy
                    gaze = (int(prev_gx), int(prev_gy))

            tx, ty, name = targets[idx]
            draw_target_dot(frame, (tx, ty), name)
            cv2.putText(frame, f"คาลิเบรต: {name}  ({idx+1}/9)",
                        (20, 40), FONT, 0.9, (255,255,255), 2)
            cv2.putText(frame, "มองจุดสีแดง แล้วคลิก 'ยืนยัน' หรือกด Space/Enter  |  F=เต็มจอ  ESC/Q=ออก",
                        (20, fh - 80), FONT, 0.7, (255,255,255), 2)

            btn_w, btn_h = 180, 55
            btn_rect = (fw - btn_w - 20, fh - btn_h - 20, fw - 20, fh - 20)
            userdata["btn_rect"] = btn_rect
            draw_button(frame, btn_rect, BTN_TEXT)

            if gaze is not None:
                cv2.circle(frame, gaze, 5, (0,255,0), -1)
                x1,y1,x2,y2 = clamp_box(gaze[0], gaze[1], fw, fh, 50, 50)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
                cv2.putText(frame, f"use:{used} gaze:{gaze}", (20, 70), FONT, 0.7, (180,255,180), 2)
            else:
                cv2.putText(frame, "กำลังหาดวงตา...", (20, 70), FONT, 0.7, (0,255,255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                set_fullscreen(fullscreen)
            if key in (32, 13):
                confirmed_click = True
            if key in (27, ord('q')):
                cap.release(); cv2.destroyAllWindows(); return

            if confirmed_click:
                confirmed_click = False
                if gaze is not None:
                    target_points.append([tx, ty])
                    gaze_points.append([gaze[0], gaze[1]])
                    idx += 1

        if len(gaze_points) >= 4:
            src = np.array(gaze_points, dtype=np.float32)
            dst = np.array(target_points, dtype=np.float32)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        else:
            H = None

        # --------- โหมดรันจริง + ทดสอบโฟกัส ---------
        # สร้าง target แรก
        fh, fw = h, w
        target_rect = rand_target_rect(fw, fh, TARGET_W, TARGET_H, margin=max(60, int(PAD_RATIO*min(fw,fh))))
        trial_idx = 1
        score = 0
        results = []
        trial_start_time = time.time()
        continuous_in_rect = 0.0   # เวลาต่อเนื่องที่ “อยู่ในกรอบ” ของรอบปัจจุบัน
        last_frame_time = time.time()

        while True:
            ok, frame = cap.read()
            if not ok: break
            if FLIP: frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            now = time.time()
            dt = now - last_frame_time
            last_frame_time = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            gaze = None
            used = "none"
            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0].landmark
                gx, gy, used = get_combined_gaze(face, fw, fh)
                if gx is not None:
                    if prev_gx is None:
                        prev_gx, prev_gy = gx, gy
                    else:
                        prev_gx = SMOOTH_ALPHA * gx + (1 - SMOOTH_ALPHA) * prev_gx
                        prev_gy = SMOOTH_ALPHA * gy + (1 - SMOOTH_ALPHA) * prev_gy
                    gaze = (int(prev_gx), int(prev_gy))

            # วาดกรอบเป้าหมาย
            x1,y1,x2,y2 = target_rect
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
            cv2.putText(frame, f"Target #{trial_idx}/{N_TRIALS}", (x1, y1-10), FONT, 0.8, (0,0,255), 2)

            # วาดกล่อง gaze (หลังคาลิเบรต)
            if gaze is not None:
                cv2.circle(frame, gaze, 4, (0,255,0), -1)
                if H is not None:
                    mx, my = perspective_map(H, gaze)
                    mx = max(0, min(mx, fw-1))
                    my = max(0, min(my, fh-1))
                    bx1,by1,bx2,by2 = clamp_box(mx, my, fw, fh, BOX_W, BOX_H)
                    cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,255,255), 2)
                    # เช็คว่าจุดคาลิเบรต (mx,my) อยู่ในกรอบเป้าหมายไหม
                    if inside_rect(mx, my, target_rect):
                        continuous_in_rect += dt
                    else:
                        continuous_in_rect = 0.0
                else:
                    # ยังไม่ได้คาลิเบรต: ใช้ gaze ดิบ (นับก็ได้แต่ความแม่นยำต่ำ)
                    bx1,by1,bx2,by2 = clamp_box(gaze[0], gaze[1], fw, fh, BOX_W, BOX_H)
                    cv2.rectangle(frame, (bx1,by1), (bx2,by2), (0,255,255), 2)
                    if inside_rect(gaze[0], gaze[1], target_rect):
                        continuous_in_rect += dt
                    else:
                        continuous_in_rect = 0.0
            else:
                cv2.putText(frame, "No eye/gaze detected", (20, 40), FONT, 0.8, (0,80,255), 2)

            # แถบความคืบหน้าความต่อเนื่อง
            bar_w = 300
            prog = np.clip(continuous_in_rect / REQUIRED_CONTINUOUS_SEC, 0.0, 1.0)
            bar_x, bar_y = 20, 80
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+20), (100,100,100), 2)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x+int(bar_w*prog), bar_y+20), (0,200,0), -1)
            cv2.putText(frame, f"Focus {continuous_in_rect:.2f}/{REQUIRED_CONTINUOUS_SEC:.0f}s", (bar_x, bar_y-10), FONT, 0.7, (180,255,180), 2)

            # เวลาสะสมของรอบ (นับตั้งแต่แสดงกรอบจน “ผ่านจริง”)
            elapsed_this_trial = now - trial_start_time
            cv2.putText(frame, f"Elapsed this trial: {elapsed_this_trial:.2f}s", (20, 120), FONT, 0.7, (255,255,255), 2)

            # คะแนนรวม
            cv2.putText(frame, f"Score: {score}", (20, 160), FONT, 0.9, (255,255,0), 2)
            cv2.putText(frame, "F=เต็มจอ  ESC/Q=ออก", (20, fh - 30), FONT, 0.8, (255,255,255), 2)

            # เงื่อนไข “ผ่าน” รอบนี้
            if continuous_in_rect >= REQUIRED_CONTINUOUS_SEC:
                score += 1
                # บันทึกผลรอบนี้: เวลากว่าจะผ่าน
                results.append([trial_idx, 1, round(elapsed_this_trial, 3)])
                # เตรียมรอบถัดไป
                trial_idx += 1
                if trial_idx > N_TRIALS:
                    # จบการทดสอบทั้งหมด
                    # สรุปผล + เซฟ CSV
                    save_results_csv(results, RESULTS_CSV)
                    cv2.putText(frame, f"FINISH! Score={score}/{N_TRIALS}", (fw//2-220, fh//2), FONT, 1.1, (0,255,0), 3)
                    y0 = fh//2 + 40
                    for i, row in enumerate(results):
                        cv2.putText(frame, f"Trial {row[0]}: {row[2]}s", (fw//2-200, y0 + i*30), FONT, 0.8, (255,255,255), 2)
                    cv2.imshow(WINDOW_NAME, frame)
                    # รอผู้ใช้กดปุ่มเพื่อออก
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key in (27, ord('q'), ord('Q')):
                            cap.release(); cv2.destroyAllWindows(); return
                else:
                    # รีเซ็ตสถานะสำหรับรอบใหม่
                    target_rect = rand_target_rect(fw, fh, TARGET_W, TARGET_H, margin=max(60, int(PAD_RATIO*min(fw,fh))))
                    trial_start_time = time.time()
                    continuous_in_rect = 0.0

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                set_fullscreen(fullscreen)
            if key in (27, ord('q')):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
