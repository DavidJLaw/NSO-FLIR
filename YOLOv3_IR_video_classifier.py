import cv2
import numpy as np

# Load YOLOv3 model
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Load COCO class names
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(1)

def detect_objects(frame, net):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    outputs = net.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #resize frame to 1080p
    frame = cv2.resize(frame, (1920, 1080))
    # Detect objects
    boxes, confidences, class_ids = detect_objects(frame, net)
    # Draw bounding boxes and labels of detected objects
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
            cv2.putText(frame, str(confidences[i]), (x, y + 60), font, 3, color, 3)
    # Display the processed frame:
    cv2.imshow("Processed OBS Window", frame)
    # Break the loop when the 'q' key is pressed:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

