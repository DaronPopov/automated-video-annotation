import os
import cv2
import numpy as np
import imageio
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display, clear_output, HTML
from ipywidgets import Output, Button, Label, FileUpload, VBox, Checkbox

# Define optimized functions
def extract_frames_imageio(video_path, output_dir, progress_output):
    progress_output.append_stdout("Extracting frames...\n")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    reader = imageio.get_reader(video_path)
    frame_files = []

    def save_frame(i, frame):
        frame_path = os.path.join(output_dir, f'frame_{i:04d}.png')
        imageio.imwrite(frame_path, frame)
        return frame_path

    with ThreadPoolExecutor() as executor:
        frame_files = list(executor.map(save_frame, range(len(reader)), reader))

    reader.close()
    progress_output.append_stdout("Frames extracted.\n")
    return frame_files

def load_frames(frame_files, progress_output):
    progress_output.append_stdout("Loading frames...\n")
    frames = []
    with ThreadPoolExecutor() as executor:
        frames = list(executor.map(cv2.imread, frame_files))
    for frame_file, frame in zip(frame_files, frames):
        if frame is None:
            progress_output.append_stdout(f"Error loading frame: {frame_file}\n")
    progress_output.append_stdout("Frames loaded.\n")
    return frames

def frames_to_3d_tensor(frames, progress_output):
    progress_output.append_stdout("Converting frames to 3D tensor...\n")
    tensor = np.stack(frames, axis=0)
    progress_output.append_stdout("3D tensor created.\n")
    return tensor

def flatten_3d_tensor(tensor, progress_output):
    progress_output.append_stdout("Flattening 3D tensor...\n")
    vectors = tensor.reshape(tensor.shape[0], -1)
    progress_output.append_stdout("3D tensor flattened.\n")
    return vectors

def process_vectors(frames, progress_output):
    progress_output.append_stdout("Processing video frames...\n")
    
    mean_brightness = []
    contrast = []
    sharpness = []
    motion_intensity = []
    
    for i, frame in enumerate(frames):
        if frame is None:
            progress_output.append_stdout(f"Error: Frame {i} is None.\n")
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Mean brightness
        mean_brightness.append(np.mean(gray_frame))
        
        # Contrast (standard deviation of pixel intensities)
        contrast.append(np.std(gray_frame))
        
        # Sharpness (using Laplacian variance)
        laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
        sharpness.append(np.var(laplacian))
        
        # Motion intensity (using frame difference)
        if i > 0:
            prev_gray_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(prev_gray_frame, gray_frame)
            motion_intensity.append(np.sum(frame_diff))
        else:
            motion_intensity.append(0)
    
    notes = {
        'mean_brightness': np.array(mean_brightness),
        'contrast': np.array(contrast),
        'sharpness': np.array(sharpness),
        'motion_intensity': np.array(motion_intensity)
    }
    
    progress_output.append_stdout("Video frames processed.\n")
    return notes

def annotate_frame_with_notes(frame, notes, frame_index):
    annotation = (
        f"Frame {frame_index}, "
        f"Mean Brightness={notes['mean_brightness'][frame_index]:.2f}, "
        f"Contrast={notes['contrast'][frame_index]:.2f}, "
        f"Sharpness={notes['sharpness'][frame_index]:.2f}, "
        f"Motion Intensity={notes['motion_intensity'][frame_index]:.2f}"
    )
    y0, dy = 30, 30  # Adjusted for 20-point font
    for i, line in enumerate(annotation.split(", ")):
        y = y0 + i * dy
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
    return frame

def reconstruct_video_with_annotations(frames, notes, output_path, fps, progress_output):
    progress_output.append_stdout("Reconstructing video with annotations...\n")
    annotated_frames = [annotate_frame_with_notes(frame, notes, i) for i, frame in enumerate(frames)]
    
    output_video = imageio.get_writer(output_path, fps=fps)
    for frame in annotated_frames:
        output_video.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    output_video.close()
    progress_output.append_stdout("Video reconstructed with annotations.\n")

def create_download_link(filepath, label):
    display(HTML(f'<a href="{filepath}" download>{label}</a>'))

# Additional Functions

def compute_color_histograms(frames, progress_output):
    progress_output.append_stdout("Computing color histograms...\n")
    histograms = []
    
    for frame in frames:
        if frame is None:
            continue
        # Compute the histogram for each color channel
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
        histograms.append((hist_b, hist_g, hist_r))
    
    progress_output.append_stdout("Color histograms computed.\n")
    return histograms

def frame_differencing(frames, progress_output):
    progress_output.append_stdout("Computing frame differences...\n")
    diff_frames = []
    prev_frame = None
    
    for frame in frames:
        if frame is None:
            continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_frame = gray_frame
            continue
        
        frame_diff = cv2.absdiff(prev_frame, gray_frame)
        diff_frames.append(frame_diff)
        prev_frame = gray_frame
    
    progress_output.append_stdout("Frame differences computed.\n")
    return diff_frames

def blur_frames(frames, progress_output):
    progress_output.append_stdout("Blurring frames...\n")
    blurred_frames = []
    
    for frame in frames:
        if frame is None:
            continue
        blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
        blurred_frames.append(blurred_frame)
    
    progress_output.append_stdout("Frames blurred.\n")
    return blurred_frames

def crop_frames(frames, crop_region, progress_output):
    progress_output.append_stdout("Cropping frames...\n")
    cropped_frames = []
    x, y, w, h = crop_region
    
    for frame in frames:
        if frame is None:
            continue
        cropped_frame = frame[y:y+h, x:x+w]
        cropped_frames.append(cropped_frame)
    
    progress_output.append_stdout("Frames cropped.\n")
    return cropped_frames

def track_objects(frames, bbox, progress_output):
    progress_output.append_stdout("Tracking objects...\n")
    tracker = cv2.TrackerCSRT_create()
    tracking_frames = []
    
    if len(frames) > 0 and frames[0] is not None:
        tracker.init(frames[0], bbox)
    
    for frame in frames:
        if frame is None:
            continue
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        tracking_frames.append(frame)
    
    progress_output.append_stdout("Object tracking completed.\n")
    return tracking_frames

def adjust_brightness(frames, brightness_factor, progress_output):
    progress_output.append_stdout("Adjusting frame brightness...\n")
    adjusted_frames = []
    
    for frame in frames:
        if frame is None:
            continue
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
        adjusted_frames.append(adjusted_frame)
    
    progress_output.append_stdout("Frame brightness adjusted.\n")
    return adjusted_frames

def detect_motion(frames, progress_output):
    progress_output.append_stdout("Detecting motion...\n")
    motion_frames = []
    prev_frame = None
    
    for frame in frames:
        if frame is None:
            continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        
        if prev_frame is None:
            prev_frame = gray_frame
            continue
        
        frame_delta = cv2.absdiff(prev_frame, gray_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        motion_frames.append(frame)
        prev_frame = gray_frame
    
    progress_output.append_stdout("Motion detection completed.\n")
    return motion_frames

def detect_faces(frames, progress_output):
    progress_output.append_stdout("Detecting faces...\n")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_frames = []
    
    for frame in frames:
        if frame is None:
            continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        face_frames.append(frame)
    
    progress_output.append_stdout("Face detection completed.\n")
    return face_frames

def compute_optical_flow(frames, progress_output):
    progress_output.append_stdout("Computing optical flow...\n")
    optical_flow_frames = []
    
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        optical_flow_frames.append(bgr)
        
        prev_gray = gray
    
    progress_output.append_stdout("Optical flow computation completed.\n")
    return optical_flow_frames

def detect_objects_yolo(frames, progress_output):
    progress_output.append_stdout("Detecting objects using YOLO...\n")
    
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"
    names_path = "coco.names"

    if not os.path.exists(weights_path) or not os.path.exists(config_path) or not os.path.exists(names_path):
        progress_output.append_stdout("Error: YOLO files are missing. Please ensure yolov3.weights, yolov3.cfg, and coco.names are in the correct directory.\n")
        return frames
    
    try:
        net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        classes = []
        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        object_detection_frames = []

        for frame_index, frame in enumerate(frames):
            if frame is None:
                continue
            progress_output.append_stdout(f"Processing frame {frame_index + 1}/{len(frames)}\n")
            
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
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

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            object_detection_frames.append(frame)

    except Exception as e:
        progress_output.append_stdout(f"Error during object detection: {str(e)}\n")
        return frames

    progress_output.append_stdout("Object detection completed.\n")
    return object_detection_frames

def stabilize_video(frames, progress_output):
    progress_output.append_stdout("Stabilizing video...\n")
    stabilized_frames = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    transforms = []
    
    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        
        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        m = cv2.estimateAffinePartial2D(prev_pts, curr_pts)[0]
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        transforms.append((dx, dy, da))
        
        prev_gray = gray
    
    for i in range(len(transforms)):
        dx, dy, da = transforms[i]
        m = np.array([[np.cos(da), -np.sin(da), dx], [np.sin(da), np.cos(da), dy]])
        frame = cv2.warpAffine(frames[i], m, (frames[i].shape[1], frames[i].shape[0]))
        stabilized_frames.append(frame)
    
    progress_output.append_stdout("Video stabilization completed.\n")
    return stabilized_frames

def detect_edges(frames, progress_output):
    progress_output.append_stdout("Detecting edges...\n")
    edge_frames = []
    
    for frame in frames:
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_frames.append(edge_frame)
    
    progress_output.append_stdout("Edge detection completed.\n")
    return edge_frames

def detect_motion_live(progress_output):
    progress_output.append_stdout("Starting real-time motion detection...\n")
    
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path
    if not cap.isOpened():
        progress_output.append_stdout("Error: Unable to open video source.\n")
        return
    
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
        
        if prev_frame is None:
            prev_frame = gray_frame
            continue
        
        frame_delta = cv2.absdiff(prev_frame, gray_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_display = buffer.tobytes()
        
        clear_output(wait=True)
        display(HTML(f'<img src="data:image/jpeg;base64,{frame_display}" width="640" height="480">'))
        
        prev_frame = gray_frame
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    progress_output.append_stdout("Real-time motion detection ended.\n")

# Create GUI elements
title = Label("Video Analysis and Annotation Tool")
upload_widget = FileUpload(accept='video/*', multiple=False)
progress_output = Output()
process_button = Button(description="Process Video", disabled=True)
start_motion_detection_button = Button(description="Start Real-Time Motion Detection")

motion_checkbox = Checkbox(description='Motion Detection')
face_detection_checkbox = Checkbox(description='Face Detection')
optical_flow_checkbox = Checkbox(description='Optical Flow Analysis')
stabilization_checkbox = Checkbox(description='Video Stabilization')
edge_detection_checkbox = Checkbox(description='Edge Detection')
color_histogram_checkbox = Checkbox(description='Color Histogram')
frame_difference_checkbox = Checkbox(description='Frame Differencing')
blur_frames_checkbox = Checkbox(description='Blur Frames')
crop_frames_checkbox = Checkbox(description='Crop Frames')
track_objects_checkbox = Checkbox(description='Track Objects')
adjust_brightness_checkbox = Checkbox(description='Adjust Brightness')

def on_upload_change(change):
    progress_output.clear_output()
    global video_path
    for filename, file_info in change['new'].items():
        video_path = os.path.join('/mnt/data', filename)
        with open(video_path, 'wb') as f:
            f.write(file_info['content'])
        progress_output.append_stdout(f"File uploaded: {filename}\n")
    process_button.disabled = False

def on_process_button_clicked(b):
    with progress_output:
        output_dir = '/mnt/data/frames'
        output_video_path = os.path.join('/mnt/data', 'annotated_video.mp4')
        
        # Execute steps
        frame_files = extract_frames_imageio(video_path, output_dir, progress_output)
        frames = load_frames(frame_files, progress_output)
        
        if stabilization_checkbox.value:
            frames = stabilize_video(frames, progress_output)
        
        tensor_3d = frames_to_3d_tensor(frames, progress_output)
        vectors = flatten_3d_tensor(tensor_3d, progress_output)
        
        # Process vectors and get notes
        notes = process_vectors(frames, progress_output)
        
        # Additional processing based on user selection
        if motion_checkbox.value:
            frames = detect_motion(frames, progress_output)
        if face_detection_checkbox.value:
            frames = detect_faces(frames, progress_output)
        if optical_flow_checkbox.value:
            frames = compute_optical_flow(frames, progress_output)
        if edge_detection_checkbox.value:
            frames = detect_edges(frames, progress_output)
        if color_histogram_checkbox.value:
            histograms = compute_color_histograms(frames, progress_output)
        if frame_difference_checkbox.value:
            diff_frames = frame_differencing(frames, progress_output)
        if blur_frames_checkbox.value:
            frames = blur_frames(frames, progress_output)
        if crop_frames_checkbox.value:
            frames = crop_frames(frames, (50, 50, 200, 200), progress_output)  # Example crop region
        if track_objects_checkbox.value:
            frames = track_objects(frames, (50, 50, 100, 100), progress_output)  # Example bbox
        if adjust_brightness_checkbox.value:
            frames = adjust_brightness(frames, 1.2, progress_output)  # Example brightness factor
        
        # Get fps of the original video
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        reader.close()

        # Reconstruct video with annotations
        reconstruct_video_with_annotations(frames, notes, output_video_path, fps, progress_output)
        
        # Provide download link
        create_download_link(output_video_path, "Download Annotated Video")

def on_start_motion_detection_button_clicked(b):
    with progress_output:
        detect_motion_live(progress_output)

upload_widget.observe(on_upload_change, names='value')
process_button.on_click(on_process_button_clicked)
start_motion_detection_button.on_click(on_start_motion_detection_button_clicked)

# Display GUI elements
display(VBox([title, upload_widget, motion_checkbox, face_detection_checkbox, optical_flow_checkbox, stabilization_checkbox, edge_detection_checkbox, color_histogram_checkbox, frame_difference_checkbox, blur_frames_checkbox, crop_frames_checkbox, track_objects_checkbox, adjust_brightness_checkbox, process_button, start_motion_detection_button, progress_output]))
