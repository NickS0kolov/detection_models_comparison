from ultralytics import RTDETR

class RTDETRProcessor:
    def __init__(self, device):
        self.device = device
        self.model = RTDETR("rtdetr-l.pt")
        self.model.to(device)                      
        
    
    def detect(self, frame):
        results = self.model.track(
            frame, 
            verbose=False, 
            classes=0, 
            conf=0.5,          
            iou=0.4,            
            max_det=100,                
            persist=True, 
            imgsz = 640,
            vid_stride = 2,
            half = True,
            device = self.device,
            tracker='processors/botsort.yaml',
        )
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                detections.append((x1, y1, x2, y2, float(conf)))
        
        return detections