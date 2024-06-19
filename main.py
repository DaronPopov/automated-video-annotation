import os
import imageio
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import kurtosis, skew
from IPython.display import display, HTML
from ipywidgets import FileUpload, Output, Button, Label

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

# Create GUI elements
title = Label("Video Analysis and Annotation Tool")
upload_widget = FileUpload(accept='video/*', multiple=False)
progress_output = Output()
process_button = Button(description="Process Video", disabled=True)

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
        tensor_3d = frames_to_3d_tensor(frames, progress_output)
        vectors = flatten_3d_tensor(tensor_3d, progress_output)
        
        # Process vectors and get notes
        notes = process_vectors(vectors, progress_output)

        # Get fps of the original video
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        reader.close()

        # Reconstruct video with annotations
        reconstruct_video_with_annotations(frames, notes, output_video_path, fps, progress_output)
        
        # Provide download link
        create_download_link(output_video_path, "Download Annotated Video")

upload_widget.observe(on_upload_change, names='value')
process_button.on_click(on_process_button_clicked)

# Display GUI elements
display(title, upload_widget, progress_output, process_button)

