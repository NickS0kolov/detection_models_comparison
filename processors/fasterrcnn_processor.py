import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

class FRCNNProcessor:
    def __init__(self, device):
        self.device = device
        
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn(weights=weights, weights_only=False)
        self.model.to(self.device)

        self.model.eval()
        
        self.person_class_id = 1
        
        self.tracker = DeepSort(
            max_age=60,
            nn_budget=100,
            embedder='mobilenet',
            half=torch.cuda.is_available(),
            bgr=False,
            embedder_gpu=torch.cuda.is_available(),
        )
        

    def detect(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
                
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if label == self.person_class_id and score > 0.8:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                detections.append(([int(x1), int(y1), int(w), int(h)], float(score), self.person_class_id))
                
        tracks = self.tracker.update_tracks(detections, frame=frame)

        results = []
        for track in tracks:
            if track.det_conf is None:
                continue
            
            conf = track.det_conf

            l, t, r, b = map(int, track.to_ltrb())
            results.append((l, t, r, b, conf))
        return results