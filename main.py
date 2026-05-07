"""
Hand Doodle - 손가락으로 그린 그림이 화면에서 살아 움직임!

조작법:
- 검지만 펴기: 펜 (그리기)
- 손바닥 활짝: 지우개
- 주먹 쥐기: 대기 (펜 떼기)
- 엄지척 👍: 그림 완성!
- SPACE: 그림 완성 (엄지척 대신 사용 가능)
- C: 캔버스만 비우기 (다시 그리기)
- X: 떠다니는 그림 모두 삭제
- Q: 종료

흐름:
1. 검지로 그림 그리기
2. 엄지척 또는 SPACE로 완성
3. 그림이 화면 가운데에 확대되어 1.5초간 표시
4. 작은 크기로 화면을 자유롭게 돌아다님
5. 캔버스 자동 초기화 → 또 다른 그림 그리기 가능
"""
import cv2
import time
import mediapipe as mp
import numpy as np

from doodle import Doodle, extract_doodle_from_canvas, get_full_size_doodle


# --- 설정 ---
DRAW_COLOR = (0, 0, 0)        # 검은색 펜
ERASE_COLOR = (255, 255, 255) # 흰색 (지우개)
BRUSH_THICKNESS = 6
ERASER_THICKNESS = 50

# 그림 완성 후 확대 표시 시간 (초)
ZOOM_DISPLAY_TIME = 1.5
# 엄지척 제스처 인식 유지 프레임 (오작동 방지)
THUMBS_UP_HOLD_FRAMES = 15


def get_fingers_open(lm_list):
    """각 손가락이 펴졌는지 판별 (엄지, 검지, 중지, 약지, 새끼)"""
    fingers = []

    # 엄지: x좌표로 판별 (오른손 기준)
    # 손목 기준 엄지 끝의 위치로 단순 판별
    if lm_list[4][1] > lm_list[3][1]:
        fingers.append(True)
    else:
        fingers.append(False)

    # 나머지: y좌표로 판별
    for tip_id in [8, 12, 16, 20]:
        if lm_list[tip_id][2] < lm_list[tip_id - 2][2]:
            fingers.append(True)
        else:
            fingers.append(False)

    return fingers


def is_thumbs_up(lm_list, fingers):
    """엄지척 제스처 판별: 엄지만 위로, 나머지는 접힘"""
    # 엄지가 손목보다 충분히 위에 있는지 확인
    thumb_above_wrist = lm_list[4][2] < lm_list[0][2] - 50

    # 검지/중지/약지/새끼는 모두 접혀있어야 함
    others_closed = (not fingers[1] and not fingers[2] and
                     not fingers[3] and not fingers[4])

    return thumb_above_wrist and others_closed


def overlay_image(background, overlay, x, y):
    """투명 배경 이미지를 다른 이미지 위에 합성"""
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # 화면 안쪽 영역만
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + ow, bw), min(y + oh, bh)
    if x1 >= x2 or y1 >= y2:
        return

    ix1, iy1 = x1 - x, y1 - y
    ix2, iy2 = ix1 + (x2 - x1), iy1 + (y2 - y1)

    roi = background[y1:y2, x1:x2]
    overlay_part = overlay[iy1:iy2, ix1:ix2]

    if overlay_part.shape[2] == 4:
        alpha = overlay_part[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (alpha * overlay_part[:, :, c] +
                            (1 - alpha) * roi[:, :, c])
    else:
        background[y1:y2, x1:x2] = overlay_part


def main():
    # MediaPipe 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # 웹캠
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 그리기 캔버스 (흰색 배경)
    canvas = np.ones((cam_h, cam_w, 3), dtype=np.uint8) * 255

    # 떠다니는 그림들
    doodles = []

    # 상태 변수
    xp, yp = 0, 0
    current_mode = "Wait"
    thumbs_up_counter = 0
    has_drawing = False  # 캔버스에 뭔가 그려져 있는지

    # 확대 연출 상태
    zoom_image = None
    zoom_start_time = 0

    print("=" * 50)
    print("Hand Doodle 시작!")
    print("=" * 50)
    print("검지: 그리기 / 손바닥: 지우개 / 주먹: 대기")
    print("엄지척 👍 또는 SPACE: 그림 완성!")
    print("C: 캔버스 비우기 / X: 떠다니는 그림 삭제 / Q: 종료")
    print("=" * 50)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # 현재 확대 연출 중인지 확인
        in_zoom_mode = zoom_image is not None

        if result.multi_hand_landmarks and not in_zoom_mode:
            for hand_lms in result.multi_hand_landmarks:
                lm_list = []
                for id, lm in enumerate(hand_lms.landmark):
                    px, py = int(lm.x * cam_w), int(lm.y * cam_h)
                    lm_list.append([id, px, py])

                if not lm_list:
                    continue

                fingers = get_fingers_open(lm_list)
                open_count = fingers.count(True)
                x1, y1 = lm_list[8][1], lm_list[8][2]

                # === 엄지척 감지 (그림 완성 트리거) ===
                if has_drawing and is_thumbs_up(lm_list, fingers):
                    thumbs_up_counter += 1
                    current_mode = f"Thumbs Up! ({thumbs_up_counter}/{THUMBS_UP_HOLD_FRAMES})"
                    # 엄지척 위치에 게이지 표시
                    tx, ty = lm_list[4][1], lm_list[4][2]
                    progress = thumbs_up_counter / THUMBS_UP_HOLD_FRAMES
                    cv2.rectangle(frame, (tx - 40, ty - 40),
                                  (tx + 40, ty - 25), (50, 50, 50), -1)
                    cv2.rectangle(frame, (tx - 40, ty - 40),
                                  (tx - 40 + int(80 * progress), ty - 25),
                                  (100, 255, 100), -1)

                    if thumbs_up_counter >= THUMBS_UP_HOLD_FRAMES:
                        # 그림 완성!
                        zoom_image = get_full_size_doodle(canvas, max_size=400)
                        zoom_start_time = time.time()
                        thumbs_up_counter = 0
                        xp, yp = 0, 0
                else:
                    thumbs_up_counter = 0

                # === 펜 모드: 검지만 펴짐 ===
                if (fingers[1] and not fingers[2] and
                        not fingers[3] and not fingers[4] and not is_thumbs_up(lm_list, fingers)):
                    current_mode = "Pen"
                    cv2.circle(frame, (x1, y1), 10, DRAW_COLOR, cv2.FILLED)

                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    cv2.line(canvas, (xp, yp), (x1, y1), DRAW_COLOR, BRUSH_THICKNESS)
                    xp, yp = x1, y1
                    has_drawing = True

                # === 지우개 모드: 손바닥 활짝 ===
                elif open_count >= 4:
                    current_mode = "Eraser"
                    center_x, center_y = lm_list[9][1], lm_list[9][2]
                    cv2.circle(frame, (center_x, center_y),
                               ERASER_THICKNESS // 2, (200, 200, 200), 2)
                    cv2.circle(canvas, (center_x, center_y),
                               ERASER_THICKNESS // 2, ERASE_COLOR, cv2.FILLED)
                    xp, yp = 0, 0

                # === 대기 모드: 주먹 ===
                elif (not fingers[1] and not fingers[2] and
                      not fingers[3] and not fingers[4]):
                    current_mode = "Wait"
                    xp, yp = 0, 0
                else:
                    current_mode = "..."
                    xp, yp = 0, 0
        else:
            # 손이 사라졌거나 zoom 모드일 때
            xp, yp = 0, 0
            if not in_zoom_mode:
                thumbs_up_counter = 0

        # === 떠다니는 그림 업데이트 ===
        for d in doodles:
            d.update()

        # === 화면 합성 ===
        # 1. 카메라 프레임에 캔버스 그림 합치기 (펜 자국)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        mask_inv_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)

        frame_no_drawing = cv2.bitwise_and(frame, cv2.bitwise_not(mask_inv_3ch))
        drawing_part = cv2.bitwise_and(canvas, mask_inv_3ch)
        frame = cv2.bitwise_or(frame_no_drawing, drawing_part)

        # 2. 떠다니는 그림 위에 그리기
        for d in doodles:
            d.draw(frame)

        # 3. 확대 연출 (그림 완성 직후)
        if in_zoom_mode:
            elapsed = time.time() - zoom_start_time

            if elapsed < ZOOM_DISPLAY_TIME:
                # 어두운 오버레이
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (cam_w, cam_h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                # 확대된 그림을 가운데에 표시 (애니메이션 효과)
                if zoom_image is not None:
                    # 0~0.3초: 커지면서 등장, 0.3~1.2초: 유지, 1.2~1.5초: 작아짐
                    if elapsed < 0.3:
                        scale = elapsed / 0.3
                    elif elapsed < ZOOM_DISPLAY_TIME - 0.3:
                        scale = 1.0
                    else:
                        scale = (ZOOM_DISPLAY_TIME - elapsed) / 0.3

                    scale = max(0.1, scale)
                    zh, zw = zoom_image.shape[:2]
                    new_w, new_h = int(zw * scale), int(zh * scale)
                    if new_w > 0 and new_h > 0:
                        scaled = cv2.resize(zoom_image, (new_w, new_h))
                        cx = (cam_w - new_w) // 2
                        cy = (cam_h - new_h) // 2
                        overlay_image(frame, scaled, cx, cy)

                # "Complete!" 텍스트
                text = "Complete!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
                tx = (cam_w - text_size[0]) // 2
                cv2.putText(frame, text, (tx, 80), font, 1.5,
                            (100, 255, 100), 3)
            else:
                # 확대 종료 → 떠다니는 그림으로 추가
                small_doodle = extract_doodle_from_canvas(canvas, target_size=150)
                if small_doodle is not None:
                    doodles.append(Doodle(small_doodle, cam_w, cam_h))

                # 캔버스 초기화
                canvas = np.ones((cam_h, cam_w, 3), dtype=np.uint8) * 255
                has_drawing = False
                zoom_image = None
                print(f"✨ 그림이 화면에 추가되었어요! (총 {len(doodles)}개)")

        # === UI 표시 ===
        cv2.putText(frame, f"Mode: {current_mode}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
        cv2.putText(frame, f"Doodles: {len(doodles)}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
        cv2.putText(frame, "Q:quit  C:clear canvas  X:clear all  SPACE:done",
                    (10, cam_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        if has_drawing and not in_zoom_mode:
            cv2.putText(frame, "Thumbs up or SPACE to finish",
                        (cam_w - 380, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 200, 100), 2)

        cv2.imshow("Hand Doodle", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.ones((cam_h, cam_w, 3), dtype=np.uint8) * 255
            has_drawing = False
            print("🧹 캔버스 비움")
        elif key == ord('x'):
            doodles = []
            print("🧹 떠다니는 그림 모두 삭제")
        elif key == ord(' ') and has_drawing and not in_zoom_mode:
            # SPACE로 완성
            zoom_image = get_full_size_doodle(canvas, max_size=400)
            zoom_start_time = time.time()
            xp, yp = 0, 0

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
