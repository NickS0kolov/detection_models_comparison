import cv2
import time
import numpy as np
import torch
import argparse
from collections import Counter
from processors.yolov11_processor import YOLO11Processor
from processors.fasterrcnn_processor import FRCNNProcessor
from processors.rtdetr_processor import RTDETRProcessor


def iou(box1, box2):
    #Считает пересечение боксов
    x1, y1, x2, y2 = box1[:4]
    x1_, y1_, x2_, y2_ = box2[:4]

    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


MODEL_ORDER = ["YOLO11", "RTDETR", "FasterRCNN"]

def normalize_models(models):
    return [m for m in MODEL_ORDER if m in models]


def merge_detections(dets_by_model, iou_thresh=0.9):
    """
    Объединяет пересекающиеся детекции от разных моделей.
    dets_by_model: dict {model_name: [(x1, y1, x2, y2, conf), ...]}
    Возвращает список объединённых детекций: (x1, y1, x2, y2, conf, models)
    """
    merged = []
    all_detections = []

    for model_name, dets in dets_by_model.items():
        for d in dets:
            all_detections.append((model_name, *d))

    used = set()

    for i, det1 in enumerate(all_detections):
        if i in used:
            continue
        model1, x1, y1, x2, y2, conf1 = det1
        group = [(x1, y1, x2, y2, conf1, [model1])]
        used.add(i)

        for j, det2 in enumerate(all_detections):
            if j in used:
                continue
            model2, x1_, y1_, x2_, y2_, conf2 = det2
            if iou((x1, y1, x2, y2, conf1), (x1_, y1_, x2_, y2_, conf2)) >= iou_thresh:
                group.append((x1_, y1_, x2_, y2_, conf2, [model2]))
                used.add(j)

        # усреднение координат и confidence
        xs1, ys1, xs2, ys2, confs, models = zip(*[(g[0], g[1], g[2], g[3], g[4], g[5]) for g in group])
        merged_box = (
            int(np.mean(xs1)),
            int(np.mean(ys1)),
            int(np.mean(xs2)),
            int(np.mean(ys2)),
            float(np.mean(confs)),
            normalize_models(set(sum(models, [])))
        )
        merged.append(merged_box)

    return merged


def blend_color(models):
    COLORS = {
        "YOLO11": np.array([0, 255, 0]),
        "FasterRCNN": np.array([0, 0, 255]),
        "RTDETR": np.array([255, 0, 0])
    }
    colors = [COLORS[m] for m in models if m in COLORS]
    if not colors:
        return (125, 125, 125)
    avg_color = np.mean(colors, axis=0)
    return tuple(map(int, avg_color))


def draw_detections(frame, detections):
    for (x1, y1, x2, y2, conf, models) in detections:
        color = blend_color(models)
        label = f"{'/'.join(models)}: {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(frame, (x1, y1 - label_height - 10),
                      (x1 + label_width, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def combination_key(models):
    return "+".join(normalize_models(models))


def main(video_path, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Входное видео: {video_path}")
    print(f"Выходное видео: {save_path}")
    print("-" * 50)

    processor_yolo = YOLO11Processor(device, model_size='m')
    processor_rcnn = FRCNNProcessor(device)
    processor_detr = RTDETRProcessor(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Разрешение: {width}x{height}, FPS: {fps:.1f}, Кадров: {total_frames}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    combo_counter = Counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = {
            "YOLO11": processor_yolo.detect(frame),
            "FasterRCNN": processor_rcnn.detect(frame),
            "RTDETR": processor_detr.detect(frame),
        }

        merged = merge_detections(detections, iou_thresh=0.8)
        frame = draw_detections(frame, merged)

        # Подсчёт комбинаций
        for _, _, _, _, _, models in merged:
            key = combination_key(models)
            combo_counter[key] += 1

        out.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Обработано: {frame_count}/{total_frames}")

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    print("\n" + "=" * 50)
    print("СТАТИСТИКА КОМБИНАЦИЙ ДЕТЕКЦИЙ:")
    for combo in [
        "YOLO11+RTDETR+FasterRCNN", "YOLO11+RTDETR", "YOLO11+FasterRCNN", "RTDETR+FasterRCNN", "YOLO11", "RTDETR", "FasterRCNN"
    ]:
        print(f"{combo:15s} = {combo_counter.get(combo, 0)}")

    print("=" * 50)
    print(f"Кадров обработано: {frame_count}")
    print(f"Время обработки: {total_time:.2f}с")
    print(f"Средний FPS: {avg_fps:.2f}")
    print(f"Выходное видео: {save_path}")
    print("=" * 50)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    main(args.video_path, args.save_path)
