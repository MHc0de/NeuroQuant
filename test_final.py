# -*- coding: utf-8 -*-

import os
import shutil
import json
import numpy as np
import nrrd
from rembg import remove
from PIL import Image, ImageDraw
import cv2
from DeepSlice import DSModel
from scipy.ndimage import map_coordinates
from skimage.measure import find_contours

def ensure_clean_dir(path):
    os.makedirs(path, exist_ok=True)
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def hex_to_rgb(hex_str):
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def build_label_color_dict(json_path):
    with open(json_path, encoding="utf-8") as f:
        struct = json.load(f)
    color_dict = {}
    stack = struct["msg"]
    while stack:
        item = stack.pop()
        if item.get("color_hex_triplet"):
            color_dict[item["id"]] = hex_to_rgb(item["color_hex_triplet"])
        stack.extend(item.get("children", []))
    return color_dict

# def remove_background_and_crop(input_path, output_path, crop_box):
#     img = Image.open(input_path).convert("RGBA")
#     bg_removed = remove(img)
#     x1, y1, x2, y2 = crop_box
#     cropped = bg_removed.crop((x1, y1, x2, y2))
#     cropped.save(output_path)
#     return crop_box

def remove_background_and_crop(input_path, output_path, crop_box):
    img = Image.open(input_path).convert("RGBA")
    bg_removed = remove(img)
    x1, y1, x2, y2 = crop_box
    
    # 이미지 크기
    W, H = bg_removed.size
    
    # 좌표 보정: 최소 0, 최대 W, H 내로 제한
    x1 = max(0, min(x1, W-1))
    y1 = max(0, min(y1, H-1))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    
    # 크롭 영역이 유효한지 확인
    if x2 <= x1 or y2 <= y1:
        print(f"[ERROR] crop_box가 유효하지 않습니다: ({x1}, {y1}, {x2}, {y2})")
        return None

    cropped = bg_removed.crop((x1, y1, x2, y2))
    cropped.save(output_path)
    return (x1, y1, x2, y2)

def paste_on_original(original, overlay_path, crop_box, output_path):
    # original: str(경로) 또는 numpy array
    # overlay_path: str
    # crop_box: (x_min, y_min, x_max, y_max)
    # output_path: str

    # original 입력이 경로인지 배열인지 판단
    if isinstance(original, str):
        original_img = Image.open(original).convert('RGBA')
    elif isinstance(original, np.ndarray):
        # OpenCV: BGR → RGB
        original_img = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGBA))
    else:
        raise ValueError("original은 파일 경로나 numpy array여야 합니다.")

    overlay_img = Image.open(overlay_path).convert('RGBA')
    x_min, y_min, x_max, y_max = crop_box
    # 로컬(크롭) 좌표 (u, v)의 글로벌(원본) 위치는 (u + x_min, v + y_min)
    # 예시: 로컬 좌표 (10, 20) → 글로벌 좌표 (10 + x_min, 20 + y_min)
    temp_img = Image.new('RGBA', original_img.size, (0, 0, 0, 0))
    temp_img.paste(overlay_img, (x_min, y_min), overlay_img)
    result = Image.alpha_composite(original_img, temp_img)
    result.save(output_path)
    return output_path

def predict_deepslice(input_dir, species='mouse'):
    model = DSModel(species)
    model.predict(input_dir, ensemble=True, section_numbers=False)
    output_path = os.path.join(input_dir, 'Deepslice_Results')
    model.save_predictions(output_path)
    return output_path

def get_anchoring(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    anchor = data["slices"][0]["anchoring"]
    width = int(data["slices"][0]["width"])
    height = int(data["slices"][0]["height"])
    return anchor, width, height

def load_and_align_annotation(annotation_path):
    annotation_data, _ = nrrd.read(annotation_path)
    annotation_data = np.rot90(annotation_data, k=1, axes=(0, 2))
    annotation_data = np.rot90(annotation_data, k=3, axes=(1, 2))
    annotation_data = np.flip(annotation_data, axis=0)
    annotation_data = np.flip(annotation_data, axis=1)
    return annotation_data

def create_slice(annotation_data, anchor, width, height):
    origin = np.array(anchor[:3])
    u_vec = np.array(anchor[3:6])
    v_vec = np.array(anchor[6:9])
    ii, jj = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    plane_points = (origin[:, None, None] +
                    (ii[None, :, :] / (width-1)) * u_vec[:, None, None] +
                    (jj[None, :, :] / (height-1)) * v_vec[:, None, None])
    plane_points = plane_points.reshape(3, -1)
    for dim in range(3):
        plane_points[dim] = np.clip(plane_points[dim], 0, annotation_data.shape[dim] - 1)
    coords = [plane_points[0], plane_points[1], plane_points[2]]
    slice_img = map_coordinates(annotation_data, coords, order=0, mode='nearest').reshape(height, width)
    return slice_img

def label_to_rgb(slice_img, label_color):
    img_rgb = np.zeros(slice_img.shape + (3,), dtype=np.uint8)
    for label, color in label_color.items():
        mask = (slice_img == label)
        img_rgb[mask] = color
    return img_rgb

def save_image(img_array, output_path):
    img = Image.fromarray(img_array)
    img.save(output_path)
    return output_path

def overlay_label(sample_img_path, label_img_path, output_path, transparency=0.8):
    sample_img = Image.open(sample_img_path).convert('RGBA')
    label_img = Image.open(label_img_path).convert('RGBA')
    if sample_img.size != label_img.size:
        label_img = label_img.resize(sample_img.size, Image.ANTIALIAS)
    label_img_np = np.array(label_img)
    black_mask = np.all(label_img_np[..., :3] == 0, axis=-1)
    label_img_np[..., 3][black_mask] = 0
    alpha_value = int(255 * (1 - transparency))
    label_img_np[..., 3][~black_mask] = alpha_value
    label_img_with_alpha = Image.fromarray(label_img_np, 'RGBA')
    overlay_img = Image.alpha_composite(sample_img, label_img_with_alpha)
    overlay_img.save(output_path)
    return output_path

def draw_label_contours(sample_img_path, label_img_path, output_path, use_label_color=True, single_color=(255, 0, 0)):
    sample_img = Image.open(sample_img_path).convert('RGBA')
    label_img_np = np.array(Image.open(label_img_path).convert('RGBA'))
    label_img_rgb = label_img_np[..., :3]
    labels, label_map = np.unique(label_img_rgb.reshape(-1, 3), axis=0, return_inverse=True)
    label_map = label_map.reshape(label_img_rgb.shape[:2])
    h, w = label_img_rgb.shape[:2]
    contour_img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(contour_img)
    for i, rgb in enumerate(labels):
        if np.all(rgb == 0):
            continue
        mask = (label_map == i).astype(np.uint8)
        contours = find_contours(mask, level=0.5)
        if use_label_color:
            edge_color = tuple([int(x) for x in rgb]) + (255,)
        else:
            edge_color = tuple(single_color) + (255,)
        for contour in contours:
            contour_xy = [(float(x), float(y)) for y, x in contour]
            if len(contour_xy) > 1:
                draw.line(contour_xy, fill=edge_color, width=2)
    if sample_img.size != contour_img.size:
        contour_img = contour_img.resize(sample_img.size, Image.ANTIALIAS)
    overlay_contour = Image.alpha_composite(sample_img, contour_img)
    overlay_contour.save(output_path)
    return output_path

def select_length_and_roi(image, window_name='Select Length and ROI', window_size=(1280, 720)):
    h, w = image.shape[:2]
    scale_w = window_size[0] / w
    scale_h = window_size[1] / h
    scale = min(scale_w, scale_h, 1.0)
    if scale < 1.0:
        disp_w, disp_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(image, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        disp_w, disp_h = w, h
        img_resized = image.copy()

    phase = "length"
    length_points = []
    length_mouse = None
    roi_box = None
    drawing = False
    ix = iy = fx = fy = -1
    mouse_x = mouse_y = -1
    roi_tmp_box = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal phase, length_points, length_mouse, drawing, ix, iy, fx, fy, roi_box, mouse_x, mouse_y, roi_tmp_box

        if phase == "length":
            if event == cv2.EVENT_LBUTTONDOWN:
                length_points.append((x, y))
                if len(length_points) == 1:
                    drawing = True
            elif event == cv2.EVENT_MOUSEMOVE:
                if len(length_points) == 1:
                    length_mouse = (x, y)  # 항상 업데이트!
            elif event == cv2.EVENT_LBUTTONUP:
                if len(length_points) == 2:
                    drawing = False
                    phase = "roi"
                    length_mouse = None
        elif phase == "roi":
            mouse_x, mouse_y = x, y
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                ix, iy = x, y
                fx, fy = x, y
                roi_tmp_box = (ix, iy, fx, fy)
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    fx, fy = x, y
                    roi_tmp_box = (ix, iy, fx, fy)
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                fx, fy = x, y
                roi_tmp_box = (ix, iy, fx, fy)
                # roi_box는 엔터로 확정

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp_disp = img_resized.copy()
        if phase == "length":
            if len(length_points) == 1 and length_mouse is not None:
                cv2.line(temp_disp, length_points[0], length_mouse, (0, 0, 255), 2)
                cv2.circle(temp_disp, length_points[0], 4, (255, 0, 0), -1)
                cv2.circle(temp_disp, length_mouse, 4, (0, 255, 0), -1)
            if len(length_points) == 2:
                cv2.line(temp_disp, length_points[0], length_points[1], (0, 0, 255), 2)
                cv2.circle(temp_disp, length_points[0], 4, (255, 0, 0), -1)
                cv2.circle(temp_disp, length_points[1], 4, (0, 255, 0), -1)
            msg = "Scale bar: Click two points with mouse"
            cv2.putText(temp_disp, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        elif phase == "roi":
            if roi_tmp_box is not None:
                x1, y1, x2, y2 = roi_tmp_box
                cv2.rectangle(temp_disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if 0 <= mouse_x < disp_w and 0 <= mouse_y < disp_h:
                cv2.line(temp_disp, (0, mouse_y), (disp_w, mouse_y), (255, 0, 0), 1)
                cv2.line(temp_disp, (mouse_x, 0), (mouse_x, disp_h), (255, 0, 0), 1)
            msg = "ROI: Drag with mouse to draw ROI (Press Enter to confirm)"
            cv2.putText(temp_disp, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        cv2.imshow(window_name, temp_disp)
        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return None, None
        if phase == "roi" and key in [13, 10]:  # Enter
            if roi_tmp_box is not None:
                x1, y1, x2, y2 = roi_tmp_box
                x1, y1 = min(x1, x2), min(y1, y2)
                x2, y2 = max(x1, x2), max(y1, y2)
                roi_box = (x1, y1, x2, y2)
                break

    cv2.destroyWindow(window_name)

    if scale < 1.0:
        length_points = [(int(x/scale), int(y/scale)) for (x, y) in length_points]
        x1, y1, x2, y2 = roi_box
        x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
        roi_box = (x1, y1, x2, y2)

    if len(length_points) == 2:
        (x1, y1), (x2, y2) = length_points
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    else:
        dist = None

    return dist, roi_box

def convert_point_by_scale(x, y, from_mag, to_mag, img_width, img_height):
    cx_img = img_width / 2
    cy_img = img_height / 2
    dx = x - cx_img
    dy = y - cy_img
    scale = to_mag / from_mag
    new_x = int(cx_img + dx * scale)
    new_y = int(cy_img + dy * scale)
    return new_x, new_y

def main():
    temp_path = r".\temp"
    temp_target_path = os.path.join(temp_path, "target")
    temp_result_path = os.path.join(temp_path, "result")
    data_path = r".\data"
    ABAMouse_struct_path = os.path.join(data_path, "ABAMouse_CCFv3_annotation_struct.json")
    annotation_path = os.path.join(data_path, "ABAMouse_CCFv3_annotation_2017_25um.nrrd")
    input_path = r".\input.png" # ❤️ 이미지 경로가 아닌 카메라 프레임을 입력받아야 합니다. 카메라 프레임에서 b버튼을 누르면 이미지를 저장하고 그걸 받아오기

    # ----- 배율/네모 크기 관련 변수 -----
    # scale_dict = {'1': 106, '4': 340, '0': 850}   # 1.25X, 4X, 10X 픽셀 크기
    # um_scale = {'1': 500/106, '4': 500/340, '0': 500/850}  # um/px
    mag_label = {'1': '1.25X', '4': '4X', '0': '10X'}
    mag_label_num = {'1': 1.25, '4': 4.0, '0': 10.0}
    current_scale = '1'   # 현재 배율 (초기값: 1.25X)
    previous_scale = '1'  # 이전 배율 (초기값: 1.25X)

    center_x, center_y = None, None  # 이미지 중심좌표
    box_sample_x_um, box_sample_y_um = 0, 0  # 샘플좌표 (um, 상대)
    fix_box = False  # 네모 고정(확정) 여부
    
    # 1. 카메라에서 프레임 받아와서 b키로 저장
    cap = cv2.VideoCapture(0)  # 번호는 환경에 맞게
    print("b: 이미지 저장 후 파이프라인 실행 / ESC: 종료")

    process_flag = False  # 파이프라인 실행 여부
    scale_flag = False  # 스케일 크기 계산 여부
    box_size_flag = False  # 네모 크기 계산 여부
    before_img = None
    before_rect = None
    box_color = (0, 255, 0) # 네모 색상 (초기값: 초록색)
    before_cx = None
    before_cy = None

    base_len_1_25x = None  # 1.25X 기준 선 길이 (픽셀)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라를 읽을 수 없습니다.")
            break
        
        # 현재 before_rect가 있으면 네모 표시
        show_frame = frame.copy()
        
        if box_size_flag:
            from_mag_x, from_mag_y, _, _ = before_rect
            # 10X 네모 정보            
            x1_10x, y1_10x = convert_point_by_scale(from_mag_x, from_mag_y, 1.25, 10.0, show_frame.shape[1], show_frame.shape[0])
            w_10x, h_10x = scale_dict[current_scale], scale_dict[current_scale]
            
            x2_10x = x1_10x + w_10x
            y2_10x = y1_10x + h_10x
            
            # 화면(프레임)을 벗어나는지 체크
            if x1_10x < 0 or y1_10x < 0 or x2_10x > show_frame.shape[1] or y2_10x > show_frame.shape[0]:
                box_color = (0, 0, 255)  # 빨간색 (범위 초과)
            else:
                box_color = (255, 0, 0)  # 파란색 (OK)
                
            box_size_flag = False  # 네모 크기 계산 완료 후 플래그 해제
            
        if scale_flag:
            from_mag_x, from_mag_y, _, _ = before_rect
            
            previous_scale_val = mag_label_num[previous_scale]  # 이전 배율의 숫자값
            current_scale_val = mag_label_num[current_scale]  # 현재 배율의 숫자값
            
            # print(f"이전 배율: {previous_scale} ({mag_label[previous_scale]})")
            # print(f"현재 배율: {current_scale} ({mag_label[current_scale]})")
            
            if current_scale == previous_scale:
                scaled_x, scaled_y = from_mag_x, from_mag_y
            else:
                scaled_x, scaled_y = convert_point_by_scale(from_mag_x, from_mag_y, previous_scale_val, current_scale_val, show_frame.shape[1], show_frame.shape[0])
            
            scaled_w, scaled_h = scale_dict[current_scale], scale_dict[current_scale]
            
            print(f"scaled_x: {scaled_x}, scaled_y: {scaled_y}, scaled_w: {scaled_w}, scaled_h: {scaled_h}")
            print(f"scaled_center: ({scaled_x + scaled_w // 2}, {scaled_y + scaled_h // 2})")

            before_rect = scaled_x, scaled_y, scaled_w, scaled_h
                        
            scale_flag = False  # 스케일 계산 완료 후 플래그 해제

        # 네모 표시 (fix_box가 True일 때만)
        if fix_box:
            box_x, box_y, box_w, box_h = before_rect
            cv2.rectangle(show_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, 2)
            # 배율(텍스트) 표시
            cv2.putText(show_frame, mag_label[current_scale], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            
        # 반드시 show_frame을 imshow 해야 함
        cv2.imshow('Camera Capture', show_frame)

        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # ESC
            break
        
        # b: before 프레임 저장 및 네모 위치 추천, rect 등 계산
        if key == ord('b'):
            cv2.imwrite(input_path, frame)
            print(f"{input_path}로 프레임을 저장하였습니다.")
            # input_path = r".\b_test.png" # ❌
            process_flag = True  # 파이프라인 실행 요청

            box_size_um = 200  # ← 한 번만 선언, 이 값만 바꾸면 네모 크기 쉽게 바꿀 수 있음

            # 스케일바 크기 계산산
            if base_len_1_25x is None:
                tmp_img = frame.copy()
                # tmp_img = cv2.imread(r".\b_test.png") # ❌
                base_len_1_25x, crop_box = select_length_and_roi(tmp_img) # base_len_1_25x : 1000um 에 해당하는 픽셀
                print(f"기준 길이(1.25X): {base_len_1_25x:.2f} px, ROI: {crop_box}")
                um_per_px_1_25x = 1000.0 / base_len_1_25x
                px_per_um_1_25x = base_len_1_25x / 1000.0
                
                scale_dict = {
                    '1': int(round(box_size_um * px_per_um_1_25x)),
                    '4': int(round(box_size_um * px_per_um_1_25x * (4 / 1.25))),
                    '0': int(round(box_size_um * px_per_um_1_25x * (10 / 1.25))),
                }
                um_scale = {k: box_size_um / v for k, v in scale_dict.items()}
                print("각 배율 별 네모 한 변 픽셀:", scale_dict)
                print("각 배율 별 um/px:", um_scale)
        
        if fix_box: # 네모가 고정된 상태에서만!
            if key == ord('1'):
                previous_scale = current_scale
                current_scale = '1'
                scale_flag = True
            elif key == ord('4'):
                previous_scale = current_scale
                current_scale = '4'
                scale_flag = True
            elif key == ord('0'):
                previous_scale = current_scale
                current_scale = '0'
                scale_flag = True
        
        # --- 네모 fix 해제/재설정 (5번, 원할 때만) ---
        if key == ord('r'):
            fix_box = False
            scale_flag = False
            base_len_1_25x = None  # 기준 길이 초기화

        # 파이프라인 한 번만 실행
        if process_flag: 
            ensure_clean_dir(temp_target_path)
            ensure_clean_dir(temp_result_path)

            label_color = build_label_color_dict(ABAMouse_struct_path)

            # None 값을 반환하는 경우(ESC 키를 눌러 선택을 취소한 경우) 프로그램 종료
            if crop_box is None:
                print("[WARN] No area selected. Program terminated. (select_roi function returned None)")
                return

            # 2. 배경 제거 및 사용자가 지정한 영역으로 크롭
            bg_removed_img_path = os.path.join(temp_target_path, "rembg_cropped.png")
            remove_background_and_crop(input_path, bg_removed_img_path, crop_box)

            # 3. DeepSlice 예측 (크롭 이미지로 실행)
            predict_deepslice(temp_target_path)

            # 4. anchoring, width, height 추출
            json_path_for_results = os.path.join(temp_target_path, 'Deepslice_Results.json')
            anchor, width, height = get_anchoring(json_path_for_results)

            # 5. annotation 데이터 로드 및 slice 생성
            annotation_data = load_and_align_annotation(annotation_path)
            slice_img = create_slice(annotation_data, anchor, width, height) # 여기서 slice_img는 색상 값이 아닌 아틀라스 id를 포함하고 있습니다.

            img_rgb = label_to_rgb(slice_img, label_color) # prelimbic area = "2FA850" # ✅ 영역 번호 변경

            # 2FA850 색상에 해당하는 영역 찾기
            target_rgb = (47, 168, 80)  # 2FA850의 RGB
            mask = np.all(img_rgb == target_rgb, axis=-1)

            # 해당 영역의 좌표 추출
            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                # 중심점 계산
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                box_size = int(round(box_size_um * px_per_um_1_25x))
                half = box_size // 2

                # 중심 기준 오른쪽에 네모 그리기
                x1 = min(cx + 1, img_rgb.shape[1] - box_size)
                x2 = x1 + box_size - 1
                y1 = max(cy - half, 0)
                y2 = min(y1 + box_size - 1, img_rgb.shape[0] - 1)

                # y1, y2 보정 (만약 box_size만큼 못 그릴 경우)
                if y2 - y1 + 1 < box_size:
                    y1 = max(0, y2 - box_size + 1)

                # x1, x2 보정 (만약 box_size만큼 못 그릴 경우)
                if x2 - x1 + 1 < box_size:
                    x1 = max(0, x2 - box_size + 1)

                # 네모 상자 그리기 (빨간색, 두께 3)
                img_rgb_on_rect = img_rgb.copy()
                cv2.rectangle(img_rgb_on_rect, (x1, y1), (x2, y2), (0, 0, 255), 3)
                before_cx = (x1 + x2) // 2
                before_cy = (y1 + y2) // 2
                before_rect = (x1, y1, x2 - x1, y2 - y1)

                # *** 중심좌표를 샘플 상대좌표로 환산 ***
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                box_sample_x_um = 0
                box_sample_y_um = 0
                fix_box = True  # 네모 고정 상태로 변경
            else:
                print("[WARN] 2FA850 영역이 이미지에 없습니다.")
                before_cx, before_cy, before_rect = None, None, None

            annotation_img_path = os.path.join(temp_result_path, "annotation.png")
            annotation_on_rect_img_path = os.path.join(temp_result_path, "annotation_on_rect.png")
            save_image(img_rgb, annotation_img_path)
            save_image(img_rgb_on_rect, annotation_on_rect_img_path)

            # 7. contour (라벨 색상) 합성, 원본에 정합
            contour_path_cropped = os.path.join(temp_result_path, 'overlay_result_contour_cropped.png')
            draw_label_contours(bg_removed_img_path, annotation_img_path, contour_path_cropped)
            contour_path = os.path.join(temp_result_path, 'overlay_result_contour.png')
            paste_on_original(frame, contour_path_cropped, crop_box, contour_path) # frame에 contour 선 그리기

            # -------------- 여기서부터 추가 --------------
            # 네모 상자(box)를 overlay_result_contour.png 위에 그리고 파일로 저장
            if before_rect is not None:
                # 1. overlay_result_contour.png 이미지를 OpenCV BGR로 불러오기
                contour_cv_img = cv2.imread(contour_path)  # shape: (H, W, 3), BGR

                # 2. 네모 좌표 (before_rect)에서 x, y, w, h 추출
                offset_x, offset_y, _, _ = crop_box
                paste_before_rect = (before_rect[0] + offset_x, before_rect[1] + offset_y, before_rect[2], before_rect[3])
                before_rect = paste_before_rect
                x, y, w, h = before_rect

                # 3. 네모 그리기 (빨간색, 두께 3)
                cv2.rectangle(contour_cv_img, (x, y), (x + w, y + h), (0, 0, 255), 3)

                # 4. 파일로 저장 (예: 'overlay_result_contour_with_box.png')
                result_with_box_path = os.path.join(temp_result_path, 'overlay_result_contour_with_box.png')
                cv2.imwrite(result_with_box_path, contour_cv_img)
                print(f"추천 네모가 포함된 이미지를 '{result_with_box_path}'에 저장하였습니다.")

            before_img = frame.copy()  # b로 저장된 프레임
            # before_img = cv2.imread(r".\b_test.png")  # ❌
            box_size_flag = True
            scale_flag = True
            process_flag = False

        # a: after 프레임 저장 및 트래킹, 그리고 before 정보 갱신
        if key == ord('a') and before_img is not None and before_rect is not None:
            after_img = frame.copy()
            print("after 이미지가 저장되었습니다. ORB 트래킹을 실행합니다.")
            # after_img = cv2.imread(r".\a_test.png") # ❌

            orb = cv2.ORB_create(5000)
            kp1, des1 = orb.detectAndCompute(before_img, None)
            kp2, des2 = orb.detectAndCompute(after_img, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) > 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                # ROI 네 꼭짓점
                x, y, w_rect, h_rect = before_rect
                rect_pts = np.float32([
                    [x, y],
                    [x + w_rect, y],
                    [x + w_rect, y + h_rect],
                    [x, y + h_rect]
                ]).reshape(-1, 1, 2)
                dst_rect_pts = cv2.transform(rect_pts, M)
                pts = np.int32(dst_rect_pts).reshape(-1, 2)

                # 트래킹 결과 네모를 현재 창에 그림
                # cv2.polylines(show_frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                # 별도의 imshow/창 닫기 없이, 기존의 show_frame을 계속 imshow('Camera Capture', show_frame)로 띄우면 됨
                
                # *** before 정보 갱신 ***
                # 네 꼭짓점 기준 min/max로 새 rect 계산
                xs, ys = pts[:, 0], pts[:, 1]
                new_x, new_y = np.min(xs), np.min(ys)
                new_w, new_h = np.max(xs) - np.min(xs), np.max(ys) - np.min(ys)
                before_rect = (int(new_x), int(new_y), int(new_w), int(new_h))
                before_img = after_img.copy()
                
                # 기존 박스 좌표 출력
                print(f"before box: (x={x}, y={y}, w={w_rect}, h={h_rect})")
                # 변경된 박스(폴리라인)의 네 꼭짓점 좌표 출력
                print(f"변환된 박스 4점 좌표: {pts}")
                print(f"after box: (x={int(new_x)}, y={int(new_y)}, w={int(new_w)}, h={int(new_h)})")
                
                
            else:
                print("특징점 매칭이 충분하지 않습니다.")

            print("Processing complete.")

            process_flag = False  # 한 번 실행 후 플래그 해제

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
