import cv2
import numpy as np
import mediapipe as mp
import time

# ---------------------- ปรับแต่งได้ ----------------------
CAM_INDEX = 0
FLIP = True
BOX_W, BOX_H = 60, 60       # ลดขนาดกล่องเพื่อเพิ่มความแม่นยำ (เดิม 100x100)
SMOOTH_ALPHA = 0.25
MIN_IRIS_AREA = 6.0         # ยิ่งสูงยิ่งคัดกรองตอนหลับตา/เบลอมากขึ้น
BTN_TEXT = "ยืนยัน"
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "Gaze Calibrate 9 points (Two Eyes)"
START_FULLSCREEN = True
PAD_RATIO = 0.08            # เว้นขอบรอบ ๆ สำหรับจุดคาลิเบรต

# เก็บหลายเฟรมตอนยืนยันคาลิเบรต
CALIB_SAMPLE_DURATION = 0.7 # วินาทีที่เก็บตัวอย่าง (แนะนำ 0.5–1.0s)
CALIB_KEEP_PERCENTILE = 80   # เก็บ 80% ที่ใกล้ median (ตัด outlier)
# ---------------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh
LEFT_IRIS  = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

def iris_center_and_area(landmarks, idx_list, w, h):
    """คืนค่า (cx, cy, area) ของไอริสจากจุด 4 จุด; ถ้าคำนวนไม่ได้ให้ area=0"""
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

def draw_target(frame, pt, label):
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
    """คืน (gaze_x, gaze_y, used) โดย:
       - ถ้าเจอทั้งสองตา (ผ่าน area) -> เฉลี่ย
       - ถ้าเจอข้างเดียว -> ใช้ข้างนั้น
       - ไม่เจอ -> (None, None, 'none')"""
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

def collect_gaze_samples(face_mesh, cap, duration=CALIB_SAMPLE_DURATION, alpha=SMOOTH_ALPHA, flip=FLIP):
    """
    เก็บตัวอย่าง gaze หลายเฟรมระยะเวลาสั้น ๆ -> median + outlier rejection
    คืน (gx, gy) แบบ int หรือ None ถ้าเก็บไม่ได้
    """
    samples = []
    t0 = time.time()
    prev = None

    while time.time() - t0 < duration:
        ok, frame = cap.read()
        if not ok: break
        if flip: frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            continue

        face = res.multi_face_landmarks[0].landmark
        gx, gy, used = get_combined_gaze(face, w, h)
        if gx is None:
            continue

        # smooth เบา ๆ ระหว่างเก็บ เพื่อกัน jitter
        if prev is None:
            prev = (gx, gy)
        else:
            prev = (alpha * gx + (1 - alpha) * prev[0],
                    alpha * gy + (1 - alpha) * prev[1])

        samples.append(prev)

    if len(samples) < 3:
        return None

    arr = np.array(samples, dtype=np.float32)
    med = np.median(arr, axis=0)

    # ตัด outlier: เก็บเฉพาะเปอร์เซ็นไทล์ที่ใกล้ median
    d = np.linalg.norm(arr - med, axis=1)
    keep_mask = d < np.percentile(d, CALIB_KEEP_PERCENTILE)
    keep = arr[keep_mask] if keep_mask.any() else arr
    final = np.median(keep, axis=0)
    return int(final[0]), int(final[1])

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("ไม่พบกล้อง/เปิดกล้องไม่ได้")

    prev_gx, prev_gy = None, None
    H = None
    confirmed_click = False
    fullscreen = START_FULLSCREEN

    # mouse callback เพื่อตรวจคลิกปุ่ม "ยืนยัน"
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

        # อ่านหนึ่งเฟรมเพื่อรู้ขนาด
        ok, frame = cap.read()
        if not ok:
            cap.release(); cv2.destroyAllWindows(); return
        if FLIP: frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # --------- นิยาม 9 จุดคาลิเบรต ---------
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

        # ---------- โหมดคาลิเบรต ----------
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

            # UI คาลิเบรต
            tx, ty, name = targets[idx]
            draw_target(frame, (tx, ty), name)
            cv2.putText(frame, f"คาลิเบรต: {name}  ({idx+1}/9)",
                        (20, 40), FONT, 0.9, (255,255,255), 2)
            cv2.putText(frame, "มองจุดสีแดง แล้วคลิก 'ยืนยัน' หรือกด Space/Enter  |  F=เต็มจอ  ESC/Q=ออก",
                        (20, fh - 80), FONT, 0.7, (255,255,255), 2)

            # ปุ่มยืนยัน
            btn_w, btn_h = 180, 55
            btn_rect = (fw - btn_w - 20, fh - btn_h - 20, fw - 20, fh - 20)
            userdata["btn_rect"] = btn_rect
            draw_button(frame, btn_rect, BTN_TEXT)

            # แสดง gaze สดช่วยเล็ง
            if gaze is not None:
                cv2.circle(frame, gaze, 5, (0,255,0), -1)
                x1,y1,x2,y2 = clamp_box(gaze[0], gaze[1], fw, fh, 50, 50)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
                cv2.putText(frame, f"use:{used} gaze:{gaze}", (20, 70), FONT, 0.7, (180,255,180), 2)
            else:
                cv2.putText(frame, "กำลังหาดวงตา...", (20, 70), FONT, 0.7, (0,255,255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            # toggle fullscreen
            if key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                set_fullscreen(fullscreen)

            # กด space/enter เพื่อยืนยัน
            if key in (32, 13):
                confirmed_click = True
            if key in (27, ord('q')):  # ยกเลิก
                cap.release(); cv2.destroyAllWindows(); return

            if confirmed_click:
                confirmed_click = False
                # >>> เก็บหลายเฟรม + median ตอนยืนยัน <<<
                sampled = collect_gaze_samples(face_mesh, cap, duration=CALIB_SAMPLE_DURATION,
                                               alpha=SMOOTH_ALPHA, flip=FLIP)
                if sampled is not None:
                    sx, sy = sampled
                    target_points.append([tx, ty])
                    gaze_points.append([sx, sy])
                    idx += 1
                # ถ้าเก็บไม่ได้ (ไม่มีตา/ไม่นิ่งพอ) → ให้กดยืนยันใหม่

        # สร้าง Homography ด้วย RANSAC (ใช้จุด >=4 จุด)
        if len(gaze_points) >= 4:
            src = np.array(gaze_points, dtype=np.float32)
            dst = np.array(target_points, dtype=np.float32)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        else:
            H = None

        # ---------- โหมดรันจริง ----------
        while True:
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

            # แสดงผล
            if gaze is not None:
                # วาดจุด gaze ดิบ
                cv2.circle(frame, gaze, 4, (0,255,0), -1)
                cv2.putText(frame, f"use:{used} raw:({gaze[0]},{gaze[1]})", (20, 40), FONT, 0.8, (180,255,180), 2)

                # ใช้ homography ถ้ามี
                if H is not None:
                    mx, my = perspective_map(H, gaze)
                    mx = max(0, min(mx, fw-1))
                    my = max(0, min(my, fh-1))
                    x1,y1,x2,y2 = clamp_box(mx, my, fw, fh, BOX_W, BOX_H)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
                    cv2.putText(frame, f"CAL:({mx},{my})  F=เต็มจอ  ESC/Q=ออก",
                                (20, 70), FONT, 0.8, (0,255,255), 2)
                else:
                    x1,y1,x2,y2 = clamp_box(gaze[0], gaze[1], fw, fh, BOX_W, BOX_H)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
                    cv2.putText(frame, "ยังไม่ได้คาลิเบรต (ใช้ gaze ดิบ)  F=เต็มจอ  ESC/Q=ออก",
                                (20, 70), FONT, 0.8, (0,255,255), 2)
            else:
                cv2.putText(frame, "No eye/gaze detected   F=เต็มจอ  ESC/Q=ออก", (20, 40), FONT, 0.9, (0,80,255), 2)

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
