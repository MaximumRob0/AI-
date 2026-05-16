import cv2
import numpy as np
import sys
import os
import configparser

MIN_AREA_PERCENT = 1.0

# Английские названия для графики (OpenCV не дружит с кириллицей)
COLOR_NAMES_EN = {
    "красный": "red",
    "оранжевый": "orange",
    "жёлтый": "yellow",
    "зелёный": "green",
    "голубой": "cyan",
    "синий": "blue",
    "фиолетовый": "violet",
    "розовый": "pink",
    "чёрный": "black",
    "серый": "gray",
    "белый": "white"
}

COLOR_RANGES = [
    ("красный",    (0, 0, 255),    [(0, 11), (170, 179)], (50, 255), (50, 255)),
    ("оранжевый",  (0, 165, 255),  [(11, 26)],            (50, 255), (50, 255)),
    ("жёлтый",     (0, 255, 255),  [(26, 36)],            (50, 255), (50, 255)),
    ("зелёный",    (0, 255, 0),    [(36, 86)],            (50, 255), (50, 255)),
    ("голубой",    (255, 255, 0),  [(86, 101)],           (50, 255), (50, 255)),
    ("синий",      (255, 0, 0),    [(101, 131)],          (50, 255), (50, 255)),
    ("фиолетовый", (255, 0, 255),  [(131, 151)],          (50, 255), (50, 255)),
    ("розовый",    (147, 112, 219),[(151, 169)],          (50, 255), (50, 255)),
]

ACHROMATIC = [
    ("чёрный", (0, 0, 0),       0, 50),
    ("серый",  (128, 128, 128), 50, 200),
    ("белый",  (255, 255, 255), 200, 256),
]

def classify_pixels(hsv_img, min_pixels):
    total = hsv_img.shape[0] * hsv_img.shape[1]
    H, S, V = cv2.split(hsv_img)
    class_map = np.zeros_like(hsv_img, dtype=np.uint8)
    info = []

    achro_mask = S < 50
    if np.any(achro_mask):
        for name, bgr, v_low, v_high in ACHROMATIC:
            mask = achro_mask & (V >= v_low) & (V < v_high)
            count = cv2.countNonZero(mask.astype(np.uint8))
            if count >= min_pixels:
                class_map[mask] = bgr
                info.append((name, bgr, 100.0 * count / total))
        chromatic_mask = ~achro_mask
    else:
        chromatic_mask = np.ones_like(S, dtype=bool)

    if np.any(chromatic_mask):
        for name, bgr, hue_ranges, (s_low, s_high), (v_low, v_high) in COLOR_RANGES:
            hue_mask = np.zeros_like(H, dtype=bool)
            for h_low, h_high in hue_ranges:
                hue_mask |= (H >= h_low) & (H < h_high)
            mask = (chromatic_mask & hue_mask &
                    (S >= s_low) & (S <= s_high) &
                    (V >= v_low) & (V <= v_high))
            count = cv2.countNonZero(mask.astype(np.uint8))
            if count >= min_pixels:
                class_map[mask] = bgr
                info.append((name, bgr, 100.0 * count / total))
    return class_map, info

def create_pie_chart(info, img_size=(600, 400)):
    """Рисует круговую диаграмму с английскими подписями (чистый OpenCV)."""
    width, height = img_size
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    if not info:
        cv2.putText(canvas, "No data", (width//4, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return canvas

    center = (160, height//2)
    radius = min(150, width//4)
    start_angle = -90

    for name, bgr, pct in info:
        angle = 360.0 * pct / 100.0
        if angle <= 0:
            continue
        end_angle = start_angle + angle
        cv2.ellipse(canvas, center, (radius, radius), 0, start_angle, end_angle, bgr, -1)
        start_angle = end_angle

    legend_x = 350
    legend_y = 60
    box_size = 15
    for i, (name, bgr, pct) in enumerate(info):
        y = legend_y + i * 25
        # Цветной квадратик
        cv2.rectangle(canvas, (legend_x, y), (legend_x+box_size, y+box_size), bgr, -1)
        cv2.rectangle(canvas, (legend_x, y), (legend_x+box_size, y+box_size), (0,0,0), 1)
        # Текст на английском
        en_name = COLOR_NAMES_EN.get(name, name)
        text = f"{en_name} {pct:.1f}%"
        cv2.putText(canvas, text, (legend_x+box_size+8, y+box_size-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    return canvas

def load_config():
    config = configparser.ConfigParser()
    config_path = "config.ini"
    if not os.path.exists(config_path):
        print(f"Ошибка: файл конфигурации '{config_path}' не найден.")
        sys.exit(1)

    config.read(config_path, encoding='utf-8')
    try:
        filename = config['DEFAULT']['File']
    except KeyError:
        print("Ошибка: в config.ini отсутствует параметр 'File'.")
        sys.exit(1)
    return filename

def main():
    img_path = load_config()

    img = cv2.imread(img_path)
    if img is None:
        print(f"Ошибка: файл '{img_path}' не найден или не является изображением.")
        sys.exit(1)

    print(f"Обрабатывается: {img_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    min_pixels = img.shape[0] * img.shape[1] * MIN_AREA_PERCENT / 100.0

    class_map, info = classify_pixels(hsv, min_pixels)

    # ---------- ТЕРМИНАЛ: русский язык ----------
    print(f"\nНайденные базовые цвета (площадь >= {MIN_AREA_PERCENT}%):")
    if not info:
        print("  ни один цвет не превысил порог")
    else:
        for name, bgr, pct in info:
            print(f"  {name:12} {pct:.1f}%")
    print("Средний оттенок (H): см. круговую диаграмму\n")

    overlay = cv2.addWeighted(img, 0.5, class_map, 0.5, 0)
    pie_img = create_pie_chart(info)

    # ---------- ОКНА: английские названия ----------
    cv2.namedWindow("Color Map", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Pie Chart", cv2.WINDOW_NORMAL)

    h, w = overlay.shape[:2]
    max_display_width = 900
    if w > max_display_width:
        scale = max_display_width / w
        new_w, new_h = int(w * scale), int(h * scale)
    else:
        new_w, new_h = w, h
    cv2.resizeWindow("Color Map", new_w, new_h)
    cv2.resizeWindow("Pie Chart", 600, 400)

    cv2.imshow("Color Map", overlay)
    cv2.imshow("Pie Chart", pie_img)

    print("Нажми любую клавишу, чтобы закрыть окна.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
