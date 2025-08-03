import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

class MediaPipeVisualizer:
    def __init__(self):
        """
        Initialize MediaPipe Visualizer with official Google drawing utilities
        """
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Crop coordinates (same as processor)
        self.CROP_COORDINATES = {
            'x_start': 4,
            'y_start': 163,
            'width': 665,
            'height': 364
        }
        
        # Pose landmark mapping (33 landmarks as per Google documentation)
        self.POSE_LANDMARKS = {
            0: 'nose',
            1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
            4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
            7: 'left_ear', 8: 'right_ear',
            9: 'mouth_left', 10: 'mouth_right',
            11: 'left_shoulder', 12: 'right_shoulder',
            13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist',
            17: 'left_pinky', 18: 'right_pinky',
            19: 'left_index', 20: 'right_index',
            21: 'left_thumb', 22: 'right_thumb',
            23: 'left_hip', 24: 'right_hip',
            25: 'left_knee', 26: 'right_knee',
            27: 'left_ankle', 28: 'right_ankle',
            29: 'left_heel', 30: 'right_heel',
            31: 'left_foot_index', 32: 'right_foot_index'
        }
    
    def create_pose_landmarks_from_csv_row(self, row: pd.Series):
        """
        Create MediaPipe pose landmarks from CSV data
        """
        landmarks = []
        
        # Extract all 33 pose landmarks
        for i in range(33):
            x_key = f'pose_{i}_x'
            y_key = f'pose_{i}_y'
            z_key = f'pose_{i}_z'
            vis_key = f'pose_{i}_visibility'
            
            if pd.notna(row.get(x_key, None)):
                # Create landmark using the correct MediaPipe class
                landmark = type('Landmark', (), {
                    'x': float(row[x_key]),
                    'y': float(row[y_key]),
                    'z': float(row.get(z_key, 0.0)),
                    'visibility': float(row.get(vis_key, 1.0))
                })()
                landmarks.append(landmark)
            else:
                # If landmark not found, create default
                landmark = type('Landmark', (), {
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0,
                    'visibility': 0.0
                })()
                landmarks.append(landmark)
        
        return landmarks
    
    def create_face_landmarks_from_csv_row(self, row: pd.Series):
        """
        Create MediaPipe face landmarks from CSV data
        """
        landmarks = []
        
        # Extract all 468 face landmarks
        for i in range(468):
            x_key = f'face_{i}_x'
            y_key = f'face_{i}_y'
            z_key = f'face_{i}_z'
            
            if pd.notna(row.get(x_key, None)):
                # Create landmark using the correct MediaPipe class
                landmark = type('Landmark', (), {
                    'x': float(row[x_key]),
                    'y': float(row[y_key]),
                    'z': float(row.get(z_key, 0.0))
                })()
                landmarks.append(landmark)
            else:
                # If landmark not found, create default
                landmark = type('Landmark', (), {
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0
                })()
                landmarks.append(landmark)
        
        return landmarks
    
    def create_hand_landmarks_from_csv_row(self, row: pd.Series, hand_idx: int):
        """
        Create MediaPipe hand landmarks from CSV data
        """
        landmarks = []
        prefix = f'hand_{hand_idx}_'
        
        # Extract all 21 hand landmarks
        for i in range(21):
            x_key = f'{prefix}{i}_x'
            y_key = f'{prefix}{i}_y'
            z_key = f'{prefix}{i}_z'
            
            if pd.notna(row.get(x_key, None)):
                # Create landmark using the correct MediaPipe class
                landmark = type('Landmark', (), {
                    'x': float(row[x_key]),
                    'y': float(row[y_key]),
                    'z': float(row.get(z_key, 0.0))
                })()
                landmarks.append(landmark)
            else:
                # If landmark not found, create default
                landmark = type('Landmark', (), {
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0
                })()
                landmarks.append(landmark)
        
        return landmarks
    
    def create_landmark_visualization_from_csv(self, video_path: str, csv_path: str, output_path: str, fps: int = 30):
        """
        Create video visualization from CSV MediaPipe landmarks
        """
        # Load CSV data
        print("Loading MediaPipe landmarks data from CSV...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} frames of MediaPipe landmark data")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {orig_fps}fps, {total_frames} frames")
        print(f"Target output FPS: {fps}")
        print(f"Crop region: x={self.CROP_COORDINATES['x_start']}, y={self.CROP_COORDINATES['y_start']}, w={self.CROP_COORDINATES['width']}, h={self.CROP_COORDINATES['height']}")
        
        # Process all frames without skipping to avoid delay
        print(f"Processing all frames to maintain synchronization")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Initialize video writer for cropped frame size - output directly to MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec for direct MP4 output
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (self.CROP_COORDINATES['width'], self.CROP_COORDINATES['height']))
        
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every frame to maintain synchronization
            print(f"Processing frame {processed_frames}/{len(df)}")
            
            # Get landmark data for this frame
            if processed_frames < len(df):
                frame_data = df.iloc[processed_frames]
                
                # Crop frame to processing region
                x, y, w, h = (self.CROP_COORDINATES['x_start'], 
                             self.CROP_COORDINATES['y_start'],
                             self.CROP_COORDINATES['width'], 
                             self.CROP_COORDINATES['height'])
                cropped_frame = frame[y:y+h, x:x+w]
                
                # Create landmarks from CSV data
                pose_landmarks = self.create_pose_landmarks_from_csv_row(frame_data)
                face_landmarks = self.create_face_landmarks_from_csv_row(frame_data)
                hand_landmarks_0 = self.create_hand_landmarks_from_csv_row(frame_data, 0)
                hand_landmarks_1 = self.create_hand_landmarks_from_csv_row(frame_data, 1)
                
                # Create landmark lists for MediaPipe drawing
                pose_landmark_list = type('PoseLandmarkList', (), {'landmark': pose_landmarks})()
                
                face_landmark_list = type('FaceLandmarkList', (), {'landmark': face_landmarks})()
                
                hand_landmark_list_0 = type('HandLandmarkList', (), {'landmark': hand_landmarks_0})()
                
                hand_landmark_list_1 = type('HandLandmarkList', (), {'landmark': hand_landmarks_1})()
                
                # Draw all landmarks with advanced styling
                annotated_frame = self.draw_all_landmarks_advanced(
                    cropped_frame, 
                    pose_landmark_list, 
                    face_landmark_list, 
                    [hand_landmark_list_0, hand_landmark_list_1]
                )
                
                # Write frame
                out.write(annotated_frame)
                processed_frames += 1
            else:
                break
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        # Verify output video
        output_cap = cv2.VideoCapture(output_path)
        output_frames = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_fps = int(output_cap.get(cv2.CAP_PROP_FPS))
        output_cap.release()
        
        print(f"\nVideo saved to {output_path}")
        print(f"Output video verification: {output_frames} frames written at {output_fps} FPS")
        
        if output_frames == 0:
            print("Warning: Output video appears to be empty!")
        else:
            print("MediaPipe visualization completed successfully!")
    
    def draw_all_landmarks_advanced(self, image, pose_landmarks, face_landmarks, hand_landmarks_list):
        """
        Draw all landmarks with direct OpenCV drawing for better compatibility
        """
        # Create a copy of the image for drawing
        annotated_image = image.copy()
        
        # Draw Pose Landmarks
        if pose_landmarks and len(pose_landmarks.landmark) > 0:
            self.draw_pose_landmarks(annotated_image, pose_landmarks.landmark)
        
        # Draw Face Mesh Landmarks
        if face_landmarks and len(face_landmarks.landmark) > 0:
            self.draw_face_landmarks(annotated_image, face_landmarks.landmark)
        
        # Draw Hand Landmarks
        for hand_landmarks in hand_landmarks_list:
            if hand_landmarks and len(hand_landmarks.landmark) > 0:
                self.draw_hand_landmarks(annotated_image, hand_landmarks.landmark)
        
        return annotated_image
    
    def draw_pose_landmarks(self, image, landmarks):
        """
        Draw pose landmarks with connections
        """
        h, w = image.shape[:2]
        
        # Define pose connections (key pairs to connect)
        pose_connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Shoulders to wrists
            (11, 23), (12, 24), (23, 24),  # Shoulders to hips
            (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
            (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
            (15, 17), (15, 19), (15, 21),  # Left hand
            (16, 18), (16, 20), (16, 22),  # Right hand
        ]
        
        # Draw connections
        for connection in pose_connections:
            start_idx, end_idx = connection
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                hasattr(landmarks[start_idx], 'x') and hasattr(landmarks[end_idx], 'x') and
                landmarks[start_idx].x > 0 and landmarks[end_idx].x > 0):
                
                start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            if hasattr(landmark, 'x') and landmark.x > 0:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
    
    def draw_face_landmarks(self, image, landmarks):
        """
        Draw detailed face mesh with triangular wireframe like Google Face Mesh
        """
        h, w = image.shape[:2]
        
        # Define face mesh triangles (simplified version of MediaPipe's FACEMESH_TESSELATION)
        # These are key triangles that form the face mesh
        face_triangles = [
            # Forehead and upper face
            (10, 338, 297), (338, 297, 332), (297, 332, 284), (332, 284, 251),
            (284, 251, 389), (251, 389, 356), (389, 356, 454), (356, 454, 323),
            (454, 323, 361), (323, 361, 288), (361, 288, 397), (288, 397, 365),
            (397, 365, 379), (365, 379, 378), (379, 378, 400), (378, 400, 377),
            (400, 377, 152), (377, 152, 148), (152, 148, 176), (148, 176, 149),
            (176, 149, 150), (149, 150, 136), (150, 136, 172), (136, 172, 58),
            (172, 58, 132), (58, 132, 93), (132, 93, 234), (93, 234, 127),
            (234, 127, 162), (127, 162, 21), (162, 21, 54), (21, 54, 103),
            (54, 103, 67), (103, 67, 109), (67, 109, 10),
            
            # Eyes
            (33, 7, 163), (7, 163, 144), (163, 144, 145), (144, 145, 153),
            (145, 153, 154), (153, 154, 155), (154, 155, 133), (155, 133, 173),
            (133, 173, 157), (173, 157, 158), (157, 158, 159), (158, 159, 160),
            (159, 160, 161), (160, 161, 246), (161, 246, 33),
            
            # Right eye
            (362, 382, 381), (382, 381, 380), (381, 380, 374), (380, 374, 373),
            (374, 373, 390), (373, 390, 249), (390, 249, 263), (249, 263, 466),
            (263, 466, 388), (466, 388, 387), (388, 387, 386), (387, 386, 385),
            (386, 385, 384), (385, 384, 398), (384, 398, 362),
            
            # Nose
            (168, 6, 197), (6, 197, 195), (197, 195, 5), (195, 5, 4),
            (5, 4, 1), (4, 1, 19), (1, 19, 94), (19, 94, 2),
            (94, 2, 164), (2, 164, 0), (164, 0, 11), (0, 11, 12),
            (11, 12, 13), (12, 13, 14), (13, 14, 15), (14, 15, 16),
            (15, 16, 17), (16, 17, 18), (17, 18, 200), (18, 200, 199),
            (200, 199, 175), (199, 175, 152), (175, 152, 377), (152, 377, 400),
            (377, 400, 378), (400, 378, 379), (378, 379, 365), (379, 365, 397),
            (365, 397, 288), (397, 288, 361), (288, 361, 323), (361, 323, 454),
            (323, 454, 356), (454, 356, 389), (356, 389, 251), (389, 251, 284),
            (251, 284, 332), (284, 332, 297), (332, 297, 338), (297, 338, 10),
            (338, 10, 109), (10, 109, 67), (109, 67, 103), (67, 103, 54),
            (103, 54, 21), (54, 21, 162), (21, 162, 127), (162, 127, 234),
            (127, 234, 93), (234, 93, 132), (93, 132, 58), (132, 58, 172),
            (58, 172, 136), (172, 136, 150), (136, 150, 149), (150, 149, 176),
            (149, 176, 148), (176, 148, 152), (148, 152, 175), (152, 175, 199),
            (175, 199, 200), (199, 200, 18), (200, 18, 17), (18, 17, 16),
            (17, 16, 15), (16, 15, 14), (15, 14, 13), (14, 13, 12),
            (13, 12, 11), (12, 11, 0), (11, 0, 164), (0, 164, 2),
            (164, 2, 94), (2, 94, 19), (94, 19, 1), (19, 1, 4),
            (1, 4, 5), (4, 5, 195), (5, 195, 197), (195, 197, 6),
            (197, 6, 168),
            
            # Mouth
            (61, 84, 17), (84, 17, 314), (17, 314, 405), (314, 405, 320),
            (405, 320, 307), (320, 307, 375), (307, 375, 321), (375, 321, 308),
            (321, 308, 324), (308, 324, 318), (324, 318, 78), (318, 78, 95),
            (78, 95, 88), (95, 88, 178), (88, 178, 87), (178, 87, 14),
            (87, 14, 317), (14, 317, 402), (317, 402, 318), (402, 318, 324),
            (318, 324, 308), (324, 308, 321), (308, 321, 375), (321, 375, 307),
            (375, 307, 320), (307, 320, 405), (320, 405, 314), (405, 314, 17),
            (314, 17, 84), (17, 84, 61), (84, 61, 146), (61, 146, 91),
            (146, 91, 181), (91, 181, 84), (181, 84, 17), (84, 17, 314),
            (17, 314, 405), (314, 405, 320), (405, 320, 307), (320, 307, 375),
            (307, 375, 321), (375, 321, 308), (321, 308, 324), (308, 324, 318),
            (324, 318, 402), (318, 402, 317), (402, 317, 14), (317, 14, 87),
            (14, 87, 178), (87, 178, 88), (178, 88, 95), (88, 95, 78),
            (95, 78, 318), (78, 318, 324), (318, 324, 308), (324, 308, 321),
            (308, 321, 375), (321, 375, 307), (375, 307, 320), (307, 320, 405),
            (320, 405, 314), (405, 314, 17), (314, 17, 84), (17, 84, 61),
            (84, 61, 146), (61, 146, 91), (146, 91, 181), (91, 181, 84),
        ]
        
        # Draw face mesh triangles
        for triangle in face_triangles:
            points = []
            for vertex_idx in triangle:
                if vertex_idx < len(landmarks) and hasattr(landmarks[vertex_idx], 'x') and landmarks[vertex_idx].x > 0:
                    x = int(landmarks[vertex_idx].x * w)
                    y = int(landmarks[vertex_idx].y * h)
                    points.append((x, y))
            
            # Draw triangle if all three points are valid
            if len(points) == 3:
                # Draw triangle edges
                cv2.line(image, points[0], points[1], (173, 216, 230), 1)  # Light blue
                cv2.line(image, points[1], points[2], (173, 216, 230), 1)
                cv2.line(image, points[2], points[0], (173, 216, 230), 1)
        
        # Draw key facial landmarks as small dots
        key_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
        
        for i in key_landmarks:
            if i < len(landmarks) and hasattr(landmarks[i], 'x') and landmarks[i].x > 0:
                x = int(landmarks[i].x * w)
                y = int(landmarks[i].y * h)
                cv2.circle(image, (x, y), 1, (255, 255, 255), -1)  # White dots
    
    def draw_hand_landmarks(self, image, landmarks):
        """
        Draw hand landmarks with connections
        """
        h, w = image.shape[:2]
        
        # Define hand connections
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        ]
        
        # Draw connections
        for connection in hand_connections:
            start_idx, end_idx = connection
            if (start_idx < len(landmarks) and end_idx < len(landmarks) and
                hasattr(landmarks[start_idx], 'x') and hasattr(landmarks[end_idx], 'x') and
                landmarks[start_idx].x > 0 and landmarks[end_idx].x > 0):
                
                start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(image, start_point, end_point, (255, 0, 255), 2)
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            if hasattr(landmark, 'x') and landmark.x > 0:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(image, (x, y), 3, (0, 255, 255), -1)

def main():
    parser = argparse.ArgumentParser(description='Create MediaPipe visualization from CSV landmarks')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('csv_path', help='Path to CSV file with MediaPipe landmarks')
    parser.add_argument('--output', '-o', help='Output video file path')
    parser.add_argument('--fps', type=int, default=30, help='Target output FPS')
    
    args = parser.parse_args()
    
    # Set default output path if not provided - simplified naming
    if args.output is None:
        video_name = Path(args.video_path).stem
        args.output = f"recreated_videos/{video_name}_mediapipe_landmarks.mp4"
    
    # Create visualizer and process
    visualizer = MediaPipeVisualizer()
    visualizer.create_landmark_visualization_from_csv(
        args.video_path, 
        args.csv_path, 
        args.output, 
        args.fps
    )

if __name__ == "__main__":
    main() 