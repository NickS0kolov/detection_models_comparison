import cv2
import time
import numpy as np
from processors.yolov11_processor import YOLO11Processor
from processors.fasterrcnn_processor import FRCNNProcessor
from processors.rtdetr_processor import RTDETRProcessor
import torch
import argparse

def draw_detections(frame, detections):
    for x1, y1, x2, y2, conf in detections:
        # Боксы
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Conf
        label = f'Person: {conf:.2f}'
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Фон для текста
        cv2.rectangle(frame, (x1, y1-label_height-10), 
                     (x1 + label_width, y1), (0, 255, 0), -1)
        
        cv2.putText(frame, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def main(model, model_size, video_path, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Запуск модели: {model}")
    print(f"Device: {device}")
    print(f"Входное видео: {video_path}")
    print(f"Выходное видео: {save_path}")
    print("-" * 50)
    
    if model == 'yolo':
        processor = YOLO11Processor(device, model_size=model_size)
    elif model == 'frcnn':
        processor = FRCNNProcessor(device)
    elif model == 'rtdetr':
        processor = RTDETRProcessor(device)
    else:
        raise ValueError("Модель должна быть 'yolo' или 'frcnn'")
    
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
    inference_times = []
    total_times = []
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.time()
        
        # ===== Время инференса модели =====
        inf_start = time.time()
        detections = processor.detect(frame)
        inf_time = time.time() - inf_start
        inference_times.append(inf_time)
        # ==================================
        
        # ===== Время отрисовки и записи =====
        frame = draw_detections(frame, detections)
        out.write(frame)
        total_time_frame = time.time() - frame_start
        total_times.append(total_time_frame)
        # ====================================
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            avg_fps_pipeline = 30 / np.mean(total_times[-30:])
            avg_fps_model = 30 / np.mean(inference_times[-30:])
            print(f"Обработано: {frame_count}/{total_frames} | "
                  f"FPS pipeline: {avg_fps_pipeline:.1f} | "
                  f"FPS модель: {avg_fps_model:.1f}")
    
    # Финальные метрики
    end_time = time.time()
    total_pipeline_time = end_time - start_time
    avg_fps_pipeline = frame_count / total_pipeline_time if total_pipeline_time > 0 else 0
    avg_fps_model = 1 / np.mean(inference_times) if inference_times else 0
    avg_inference_time = np.mean(inference_times) * 1000  # мс/кадр
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ:")
    print(f"Всего кадров: {frame_count}")
    print(f"Общее время пайплайна: {total_pipeline_time:.2f}с")
    print(f"Средний FPS пайплайна: {avg_fps_pipeline:.2f}")
    print(f"Средний FPS модели: {avg_fps_model:.2f}")
    print(f"Время инференса модели: {avg_inference_time:.1f}мс/кадр")
    print(f"Выходное видео: {save_path}")
    print("="*50)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo", choices=["yolo", "frcnn", "rtdetr"])
    parser.add_argument("--model_size", type=str, default="m")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    main(args.model, args.model_size, args.video_path, args.save_path)