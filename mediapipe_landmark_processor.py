import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class LandmarkData:
    """Data class to store landmark information"""
    x: float
    y: float
    z: float
    visibility: float = 1.0

class MediaPipeLandmarkProcessor:
    def __init__(self, 
                 pose_model_path: str = None,
                 face_model_path: str = None,
                 hand_model_path: str = None,
                 num_poses: int = 1,
                 num_faces: int = 1,
                 num_hands: int = 2,
                 min_pose_detection_confidence: float = 0.5,
                 min_pose_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 min_face_detection_confidence: float = 0.5,
                 min_hand_detection_confidence: float = 0.5,
                 refine_landmarks: bool = True):
        """
        Initialize MediaPipe Landmark Processor with official Google models
        
        Args:
            pose_model_path: Path to pose landmarker model (.task file)
            face_model_path: Path to face landmarker model (.task file)
            hand_model_path: Path to hand landmarker model (.task file)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Pose Landmarker
        self.pose_landmarker = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0, 1, or 2
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=True,
            min_detection_confidence=min_pose_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_face_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=num_hands,
            model_complexity=1,
            min_detection_confidence=min_hand_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Crop coordinates for processing
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
        
        # Hand landmark mapping (21 landmarks per hand)
        self.HAND_LANDMARKS = {
            0: 'wrist',
            1: 'thumb_cmc', 2: 'thumb_mcp', 3: 'thumb_ip', 4: 'thumb_tip',
            5: 'index_finger_mcp', 6: 'index_finger_pip', 7: 'index_finger_dip', 8: 'index_finger_tip',
            9: 'middle_finger_mcp', 10: 'middle_finger_pip', 11: 'middle_finger_dip', 12: 'middle_finger_tip',
            13: 'ring_finger_mcp', 14: 'ring_finger_pip', 15: 'ring_finger_dip', 16: 'ring_finger_tip',
            17: 'pinky_mcp', 18: 'pinky_pip', 19: 'pinky_dip', 20: 'pinky_tip'
        }
    
    def process_video(self, video_path: str, output_csv_path: str = None) -> str:
        """
        Process video and extract all landmarks (pose, face, hands)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"Crop region: x={self.CROP_COORDINATES['x_start']}, y={self.CROP_COORDINATES['y_start']}, w={self.CROP_COORDINATES['width']}, h={self.CROP_COORDINATES['height']}")
        
        # Prepare CSV data storage
        landmarks_data = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Crop frame to processing region
            x, y, w, h = (self.CROP_COORDINATES['x_start'], 
                         self.CROP_COORDINATES['y_start'],
                         self.CROP_COORDINATES['width'], 
                         self.CROP_COORDINATES['height'])
            cropped_frame = frame[y:y+h, x:x+w]
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with all models
            frame_data = {'frame': frame_count}
            
            # Process Pose Landmarks
            pose_results = self.pose_landmarker.process(rgb_frame)
            if pose_results.pose_landmarks:
                for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                    frame_data[f'pose_{i}_x'] = landmark.x
                    frame_data[f'pose_{i}_y'] = landmark.y
                    frame_data[f'pose_{i}_z'] = landmark.z
                    frame_data[f'pose_{i}_visibility'] = landmark.visibility
                print(f"Frame {frame_count}: Extracted {len(pose_results.pose_landmarks.landmark)} pose landmarks")
            else:
                # No pose detected, fill with NaN
                for i in range(33):
                    frame_data[f'pose_{i}_x'] = np.nan
                    frame_data[f'pose_{i}_y'] = np.nan
                    frame_data[f'pose_{i}_z'] = np.nan
                    frame_data[f'pose_{i}_visibility'] = np.nan
                print(f"Frame {frame_count}: No pose detected")
            
            # Process Face Mesh Landmarks
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                # Process first face
                face_landmarks = face_results.multi_face_landmarks[0]
                for i, landmark in enumerate(face_landmarks.landmark):
                    frame_data[f'face_{i}_x'] = landmark.x
                    frame_data[f'face_{i}_y'] = landmark.y
                    frame_data[f'face_{i}_z'] = landmark.z
                print(f"Frame {frame_count}: Extracted {len(face_landmarks.landmark)} face landmarks")
            else:
                # No face detected, fill with NaN for all 478 landmarks
                for i in range(478):  # Updated to match the actual number of landmarks
                    frame_data[f'face_{i}_x'] = np.nan
                    frame_data[f'face_{i}_y'] = np.nan
                    frame_data[f'face_{i}_z'] = np.nan
                print(f"Frame {frame_count}: No face detected")
            
            # Process Hand Landmarks
            hands_results = self.hands.process(rgb_frame)
            if hands_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                    prefix = f'hand_{hand_idx}_'
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        frame_data[f'{prefix}{i}_x'] = landmark.x
                        frame_data[f'{prefix}{i}_y'] = landmark.y
                        frame_data[f'{prefix}{i}_z'] = landmark.z
                    print(f"Frame {frame_count}: Extracted {len(hand_landmarks.landmark)} landmarks for hand {hand_idx}")
            else:
                # No hands detected, fill with NaN for both hands
                for hand_idx in range(2):
                    prefix = f'hand_{hand_idx}_'
                    for i in range(21):
                        frame_data[f'{prefix}{i}_x'] = np.nan
                        frame_data[f'{prefix}{i}_y'] = np.nan
                        frame_data[f'{prefix}{i}_z'] = np.nan
                print(f"Frame {frame_count}: No hands detected")
            
            landmarks_data.append(frame_data)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        self.pose_landmarker.close()
        self.face_mesh.close()
        self.hands.close()
        
        # Save to CSV
        if output_csv_path is None:
            video_name = Path(video_path).stem
            output_csv_path = f"csv_data/{video_name}_mediapipe_landmarks.csv"
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(landmarks_data)
        df.to_csv(output_csv_path, index=False)
        
        print(f"MediaPipe landmarks saved to: {output_csv_path}")
        print(f"Total frames processed: {len(landmarks_data)}")
        
        return output_csv_path

def main():
    parser = argparse.ArgumentParser(description='Extract MediaPipe landmarks from video using official Google models')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--num_poses', type=int, default=1, help='Maximum number of poses to detect')
    parser.add_argument('--num_faces', type=int, default=1, help='Maximum number of faces to detect')
    parser.add_argument('--num_hands', type=int, default=2, help='Maximum number of hands to detect')
    parser.add_argument('--min_pose_detection_confidence', type=float, default=0.5, help='Minimum pose detection confidence')
    parser.add_argument('--min_face_detection_confidence', type=float, default=0.5, help='Minimum face detection confidence')
    parser.add_argument('--min_hand_detection_confidence', type=float, default=0.5, help='Minimum hand detection confidence')
    parser.add_argument('--min_tracking_confidence', type=float, default=0.5, help='Minimum tracking confidence')
    parser.add_argument('--refine_landmarks', action='store_true', default=True, help='Refine landmarks around eyes and lips')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = MediaPipeLandmarkProcessor(
        num_poses=args.num_poses,
        num_faces=args.num_faces,
        num_hands=args.num_hands,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_face_detection_confidence=args.min_face_detection_confidence,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        refine_landmarks=args.refine_landmarks
    )
    
    # Process video
    output_path = processor.process_video(args.video_path, args.output)
    
    if output_path:
        print(f"MediaPipe landmark processing completed successfully!")
        print(f"Output CSV: {output_path}")
    else:
        print("MediaPipe landmark processing failed!")

if __name__ == "__main__":
    main() 