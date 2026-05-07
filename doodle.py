"""
완성된 낙서가 화면에서 자유롭게 돌아다니는 클래스
"""
import random
import math
import cv2
import numpy as np


class Doodle:
    def __init__(self, image, screen_width, screen_height):
        """
        image: BGRA 형식의 그림 이미지 (배경은 투명)
        """
        self.image = image
        self.screen_w = screen_width
        self.screen_h = screen_height

        h, w = image.shape[:2]
        # 화면 안쪽 랜덤 위치에서 시작
        self.x = float(random.randint(50, max(51, screen_width - w - 50)))
        self.y = float(random.randint(50, max(51, screen_height - h - 50)))

        # 랜덤한 속도와 방향
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2.0, 4.0)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

        # 둥실둥실 흔들리는 효과용
        self.wobble = random.uniform(0, 2 * math.pi)
        # 살짝 회전하는 효과용
        self.rotation = 0
        self.rot_speed = random.uniform(-1.5, 1.5)

    def update(self):
        """위치 갱신"""
        self.x += self.vx
        self.y += self.vy

        # 부드러운 흔들림
        self.wobble += 0.1
        self.y += math.sin(self.wobble) * 0.4

        # 회전
        self.rotation += self.rot_speed

        # 가장자리 닿으면 튕김
        h, w = self.image.shape[:2]
        if self.x < 0:
            self.x = 0
            self.vx *= -1
        elif self.x + w > self.screen_w:
            self.x = self.screen_w - w
            self.vx *= -1

        if self.y < 0:
            self.y = 0
            self.vy *= -1
        elif self.y + h > self.screen_h:
            self.y = self.screen_h - h
            self.vy *= -1

    def get_rotated_image(self):
        """회전된 이미지 반환"""
        h, w = self.image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, self.rotation, 1.0)
        rotated = cv2.warpAffine(
            self.image, matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        return rotated

    def draw(self, frame):
        """프레임 위에 낙서 그리기 (투명도 처리)"""
        img = self.get_rotated_image()
        h, w = img.shape[:2]
        x, y = int(self.x), int(self.y)

        # 화면 밖이면 스킵
        if x + w <= 0 or y + h <= 0 or x >= self.screen_w or y >= self.screen_h:
            return

        # 화면 안쪽 영역만 잘라내기
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, self.screen_w), min(y + h, self.screen_h)
        ix1, iy1 = x1 - x, y1 - y
        ix2, iy2 = ix1 + (x2 - x1), iy1 + (y2 - y1)

        roi = frame[y1:y2, x1:x2]
        doodle_part = img[iy1:iy2, ix1:ix2]

        if doodle_part.shape[2] == 4:  # 알파 채널 있음
            alpha = doodle_part[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = (alpha * doodle_part[:, :, c] +
                                (1 - alpha) * roi[:, :, c])
        else:
            frame[y1:y2, x1:x2] = doodle_part


def extract_doodle_from_canvas(canvas, target_size=120):
    """
    흰색 배경의 캔버스에서 그림 부분만 잘라내서 투명 배경 PNG로 변환
    target_size: 가장 긴 변의 픽셀 크기 (작은 크기로 리사이즈)
    """
    # 회색조 변환 후, 흰색이 아닌 픽셀(=그림) 찾기
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    # mask: 그림 부분이 흰색(255), 배경이 검정(0)

    # 그림이 있는 영역의 바운딩 박스 찾기
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None  # 그림이 없음

    x, y, w, h = cv2.boundingRect(coords)
    # 약간의 여백 추가
    padding = 20
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + padding * 2, canvas.shape[1] - x)
    h = min(h + padding * 2, canvas.shape[0] - y)

    cropped_canvas = canvas[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]

    # BGRA로 변환 (알파 채널 추가)
    bgra = cv2.cvtColor(cropped_canvas, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = cropped_mask  # 그림 부분만 보이게

    # target_size에 맞춰 리사이즈
    ch, cw = bgra.shape[:2]
    if max(ch, cw) > target_size:
        scale = target_size / max(ch, cw)
        new_w, new_h = int(cw * scale), int(ch * scale)
        bgra = cv2.resize(bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return bgra


def get_full_size_doodle(canvas, max_size=400):
    """
    확대 연출용: 그림을 크게 보여줄 때 사용 (배경 투명)
    """
    return extract_doodle_from_canvas(canvas, target_size=max_size)
