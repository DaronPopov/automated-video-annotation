import os
import imageio
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import kurtosis, skew
from IPython.display import display, HTML
from ipywidgets import FileUpload, Output, Button, Label, Checkbox, VBox, HBox, IntProgress

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
    with ThreadPoolExecutor() as executor:
        frames = list(executor.map(cv2.imread, frame_files))
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

def process_vectors(vectors, progress_output):
    progress_output.append_stdout("Processing vectors...\n")
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(np.mean, vectors, axis=1): 'mean',
            executor.submit(np.var, vectors, axis=1): 'variance',
            executor.submit(np.max, vectors, axis=1): 'max',
            executor.submit(np.min, vectors, axis=1): 'min',
            executor.submit(np.median, vectors, axis=1): 'median',
            executor.submit(np.std, vectors, axis=1): 'std_dev',
            executor.submit(np.sum, vectors, axis=1): 'sum',
            executor.submit(np.ptp, vectors, axis=1): 'range',
            executor.submit(kurtosis, vectors, axis=1): 'kurtosis',
            executor.submit(skew, vectors, axis=1): 'skewness',
            executor.submit(np.percentile, vectors, 25, axis=1): 'q1',
            executor.submit(np.percentile, vectors, 75, axis=1): 'q3',
            executor.submit(np.sum, vectors**2, axis=1): 'energy'
        }
        results = {name: future.result() for future, name in futures.items()}
    
    results['iqr'] = results['q3'] - results['q1']
    results['entropy'] = [-np.sum(p * np.log2(p)) for p in (frame/np.sum(frame) for frame in vectors)]
    results['zero_crossing_rate'] = [((frame[:-1] * frame[1:]) < 0).sum() for frame in vectors]

    progress_output.append_stdout("Vectors processed.\n")
    return results

def annotate_frame_with_notes(frame, notes, frame_index):
    height, width, _ = frame.shape
    annotation = (
        f"Frame {frame_index}, "
        f"Mean={notes['mean'][frame_index]:.2f}, "
        f"Variance={notes['variance'][frame_index]:.2f}, "
        f"Max={notes['max'][frame_index]:.2f}, Min={notes['min'][frame_index]:.2f}, "
        f"Median={notes['median'][frame_index]:.2f}, StdDev={notes['std_dev'][frame_index]:.2f}, "
        f"Sum={notes['sum'][frame_index]:.2f}, Range={notes['range'][frame_index]:.2f}, "
        f"Kurtosis={notes['kurtosis'][frame_index]:.2f}, Skewness={notes['skewness'][frame_index]:.2f}, "
        f"Q1={notes['q1'][frame_index]:.2f}, Q3={notes['q3'][frame_index]:.2f}, IQR={notes['iqr'][frame_index]:.2f}, "
        f"Energy={notes['energy'][frame_index]:.2f}, Entropy={notes['entropy'][frame_index]:.2f}, "
        f"ZeroCrossRate={notes['zero_crossing_rate'][frame_index]}"
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

def detect_motion(frames, progress_output):
    progress_output.append_stdout("Detecting motion...\n")
    motion_frames = []
    prev_frame = None
    
    for frame in frames:
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
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        classes = []
        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]

        object_detection_frames = []

        for frame in frames:
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_frames.append(edge_frame)
    
    progress_output.append_stdout("Edge detection completed.\n")
    return edge_frames

# Create GUI elements
title = Label("Video Analysis and Annotation Tool")
upload_widget = FileUpload(accept='video/*', multiple=False)
progress_output = Output()
process_button = Button(description="Process Video", disabled=True)

motion_checkbox = Checkbox(description='Motion Detection')
optical_flow_checkbox = Checkbox(description='Optical Flow Analysis')
stabilization_checkbox = Checkbox(description='Video Stabilization')
edge_detection_checkbox = Checkbox(description='Edge Detection')

# Create progress bar
progress_bar = IntProgress(value=0, min=0, max=100, description='Progress:', bar_style='info', style={'bar_color': 'gray'})
progress_output = Output(layout={'border': '1px solid black', 'width': '100%', 'height': '300px'})

# Apply greyscale CSS styling
css = """
<style>
    .widget-label {
        color: #333;
    }
    .widget-button {
        background-color: #ccc;
        border: none;
        color: #333;
        padding: 8px 16px;
        font-size: 14px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
        transition-duration: 0.4s;
    }
    .widget-button:hover {
        background-color: #666;
        color: white;
    }
    .widget-checkbox {
        color: #333;
    }
    .widget-output {
        background-color: #f9f9f9;
        color: #333;
        padding: 10px;
        font-family: 'Courier New', Courier, monospace;
    }
    .progress-bar-container {
        display: flex;
        align-items: center;
    }
    .progress-bar-container > * {
        margin-right: 10px;
    }
</style>
"""
display(HTML(css))

def on_upload_change(change):
    progress_output.clear_output()
    global video_path
    for filename, file_info in change['new'].items():
        video_path = os.path.join('/mnt/data', filename)
        with open(video_path, 'wb') as f:
            f.write(file_info['content'])
        progress_output.append_stdout(f"File uploaded: {filename}\n")
    process_button.disabled = False

def update_progress(progress, total, description):
    progress_bar.value = int((progress / total) * 100)
    progress_bar.description = description

def on_process_button_clicked(b):
    with progress_output:
        output_dir = '/mnt/data/frames'
        output_video_path = os.path.join('/mnt/data', 'annotated_video.mp4')
        
        # Execute steps
        update_progress(0, 10, 'Extracting frames')
        frame_files = extract_frames_imageio(video_path, output_dir, progress_output)
        update_progress(1, 10, 'Loading frames')
        frames = load_frames(frame_files, progress_output)
        
        if stabilization_checkbox.value:
            update_progress(2, 10, 'Stabilizing video')
            frames = stabilize_video(frames, progress_output)
        
        update_progress(3, 10, 'Converting to 3D tensor')
        tensor_3d = frames_to_3d_tensor(frames, progress_output)
        update_progress(4, 10, 'Flattening 3D tensor')
        vectors = flatten_3d_tensor(tensor_3d, progress_output)
        
        update_progress(5, 10, 'Processing vectors')
        notes = process_vectors(vectors, progress_output)
        
        if motion_checkbox.value:
            update_progress(6, 10, 'Detecting motion')
            frames = detect_motion(frames, progress_output)
        if optical_flow_checkbox.value:
            update_progress(7, 10, 'Computing optical flow')
            frames = compute_optical_flow(frames, progress_output)
        if edge_detection_checkbox.value:
            update_progress(8, 10, 'Detecting edges')
            frames = detect_edges(frames, progress_output)
        
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        reader.close()
        
        update_progress(9, 10, 'Reconstructing video')
        reconstruct_video_with_annotations(frames, notes, output_video_path, fps, progress_output)
        
        update_progress(10, 10, 'Done')
        create_download_link(output_video_path, "Download Annotated Video")

upload_widget.observe(on_upload_change, names='value')
process_button.on_click(on_process_button_clicked)

# Display GUI elements
display(VBox([title, upload_widget, HBox([motion_checkbox, optical_flow_checkbox, stabilization_checkbox, edge_detection_checkbox]), process_button, progress_bar, progress_output]))

