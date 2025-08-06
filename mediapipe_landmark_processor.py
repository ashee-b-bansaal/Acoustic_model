import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import argparse
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from tqdm import tqdm

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
            
            try:
                # Process Pose Landmarks
                pose_results = self.pose_landmarker.process(rgb_frame)
                if pose_results.pose_landmarks and pose_results.pose_landmarks.landmark:
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
                if face_results.multi_face_landmarks and len(face_results.multi_face_landmarks) > 0:
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
                if hands_results.multi_hand_landmarks and len(hands_results.multi_hand_landmarks) > 0:
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
                    
            except Exception as e:
                print(f"Frame {frame_count}: Error processing landmarks - {e}")
                # Fill all landmarks with NaN on error
                for i in range(33):
                    frame_data[f'pose_{i}_x'] = np.nan
                    frame_data[f'pose_{i}_y'] = np.nan
                    frame_data[f'pose_{i}_z'] = np.nan
                    frame_data[f'pose_{i}_visibility'] = np.nan
                for i in range(478):
                    frame_data[f'face_{i}_x'] = np.nan
                    frame_data[f'face_{i}_y'] = np.nan
                    frame_data[f'face_{i}_z'] = np.nan
                for hand_idx in range(2):
                    prefix = f'hand_{hand_idx}_'
                    for i in range(21):
                        frame_data[f'{prefix}{i}_x'] = np.nan
                        frame_data[f'{prefix}{i}_y'] = np.nan
                        frame_data[f'{prefix}{i}_z'] = np.nan
            
            landmarks_data.append(frame_data)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        
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
    
    def find_video_files(self, videoset_path: str) -> List[Dict[str, str]]:
        """Find all video files in the videoset directory."""
        video_files = []
        
        if not os.path.exists(videoset_path):
            logging.error(f"Videoset path does not exist: {videoset_path}")
            return video_files
        
        # Walk through all session folders
        for session_dir in os.listdir(videoset_path):
            session_path = os.path.join(videoset_path, session_dir)
            
            if os.path.isdir(session_path) and session_dir.startswith('session_'):
                clips_path = os.path.join(session_path, 'clips')
                
                if os.path.exists(clips_path):
                    # Find all video files in clips folder
                    for video_file in os.listdir(clips_path):
                        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            video_path = os.path.join(clips_path, video_file)
                            video_files.append({
                                'video_path': video_path,
                                'session_dir': session_dir,
                                'video_name': video_file,
                                'csv_name': f"csv_{video_file.replace('.mp4', '').replace('.avi', '').replace('.mov', '').replace('.mkv', '')}.csv"
                            })
        
        logging.info(f"Found {len(video_files)} video files to process")
        return video_files
    
    def create_csv_directory(self, dataset_path: str, session_dir: str) -> str:
        """Create CSV directory for a session if it doesn't exist."""
        session_csv_path = os.path.join(dataset_path, session_dir, 'csv')
        os.makedirs(session_csv_path, exist_ok=True)
        return session_csv_path
    
    def process_all_videos(self, dataset_path: str, force_reprocess: bool = False) -> Dict[str, int]:
        """Process all videos in the dataset and generate CSV files."""
        logging.info("Starting batch video processing...")
        
        # Validate dataset structure
        videoset_path = dataset_path.replace('/dataset', '/videoset')
        
        if not os.path.exists(dataset_path):
            logging.error(f"Dataset path does not exist: {dataset_path}")
            return {'processed': 0, 'skipped': 0, 'errors': 0}
        
        if not os.path.exists(videoset_path):
            logging.error(f"Videoset path does not exist: {videoset_path}")
            return {'processed': 0, 'skipped': 0, 'errors': 0}
        
        # Find all video files
        video_files = self.find_video_files(videoset_path)
        
        if not video_files:
            logging.error("No video files found to process")
            return {'processed': 0, 'skipped': 0, 'errors': 0}
        
        # Process videos
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # Create progress bar
        pbar = tqdm(video_files, desc="Processing videos")
        
        for video_info in pbar:
            video_path = video_info['video_path']
            session_dir = video_info['session_dir']
            csv_name = video_info['csv_name']
            
            # Create CSV directory for this session
            csv_dir = self.create_csv_directory(dataset_path, session_dir)
            csv_path = os.path.join(csv_dir, csv_name)
            
            # Update progress bar description
            pbar.set_description(f"Processing {video_info['video_name']}")
            
            try:
                # Check if CSV already exists
                if os.path.exists(csv_path) and not force_reprocess:
                    logging.info(f"CSV already exists, skipping: {csv_path}")
                    skipped_count += 1
                    continue
                
                # Process video with MediaPipe
                logging.info(f"Processing: {video_path} -> {csv_path}")
                self.process_video(video_path, csv_path)
                
                # Verify CSV was created
                if os.path.exists(csv_path):
                    processed_count += 1
                    logging.info(f"Successfully processed: {video_info['video_name']}")
                else:
                    error_count += 1
                    logging.error(f"Failed to create CSV: {csv_path}")
                    
            except Exception as e:
                error_count += 1
                logging.error(f"Error processing {video_info['video_name']}: {e}")
        
        pbar.close()
        
        # Print summary
        logging.info("="*50)
        logging.info("BATCH PROCESSING SUMMARY")
        logging.info("="*50)
        logging.info(f"Total videos found: {len(video_files)}")
        logging.info(f"Successfully processed: {processed_count}")
        logging.info(f"Skipped (already exists): {skipped_count}")
        logging.info(f"Errors: {error_count}")
        logging.info("="*50)
        
        # Clean up MediaPipe models
        try:
            self.pose_landmarker.close()
            self.face_mesh.close()
            self.hands.close()
            logging.info("MediaPipe models closed successfully")
        except Exception as e:
            logging.warning(f"Error closing MediaPipe models: {e}")
        
        return {'processed': processed_count, 'skipped': skipped_count, 'errors': error_count}

def main():
    parser = argparse.ArgumentParser(description='Extract MediaPipe landmarks from video using official Google models')
    parser.add_argument('--video_path', help='Path to input video file (for single video processing)')
    parser.add_argument('--dataset_path', help='Path to dataset folder for batch processing (e.g., /data/asl_dataset/confusing_word)')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--num_poses', type=int, default=1, help='Maximum number of poses to detect')
    parser.add_argument('--num_faces', type=int, default=1, help='Maximum number of faces to detect')
    parser.add_argument('--num_hands', type=int, default=2, help='Maximum number of hands to detect')
    parser.add_argument('--min_pose_detection_confidence', type=float, default=0.5, help='Minimum pose detection confidence')
    parser.add_argument('--min_face_detection_confidence', type=float, default=0.5, help='Minimum face detection confidence')
    parser.add_argument('--min_hand_detection_confidence', type=float, default=0.5, help='Minimum hand detection confidence')
    parser.add_argument('--min_tracking_confidence', type=float, default=0.5, help='Minimum tracking confidence')
    parser.add_argument('--refine_landmarks', action='store_true', default=True, help='Refine landmarks around eyes and lips')
    parser.add_argument('--force_reprocess', action='store_true', help='Reprocess videos even if CSV already exists')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('mediapipe_processing.log'),
            logging.StreamHandler()
        ]
    )
    
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
    
    # Check if batch processing or single video processing
    if args.dataset_path:
        # Batch processing
        logging.info("="*50)
        logging.info("BATCH VIDEO PROCESSING")
        logging.info("="*50)
        logging.info(f"Dataset path: {args.dataset_path}")
        logging.info(f"Force reprocess: {args.force_reprocess}")
        logging.info("="*50)
        
        # Validate dataset structure
        dataset_path = os.path.join(args.dataset_path, 'dataset')
        videoset_path = os.path.join(args.dataset_path, 'videoset')
        
        if not os.path.exists(dataset_path):
            logging.error(f"Dataset path does not exist: {dataset_path}")
            return
        
        if not os.path.exists(videoset_path):
            logging.error(f"Videoset path does not exist: {videoset_path}")
            return
        
        # Process all videos
        start_time = time.time()
        results = processor.process_all_videos(dataset_path, args.force_reprocess)
        end_time = time.time()
        
        # Print summary
        logging.info("="*50)
        logging.info("BATCH PROCESSING COMPLETED")
        logging.info("="*50)
        logging.info(f"Total time: {end_time - start_time:.2f} seconds")
        logging.info(f"Successfully processed: {results['processed']}")
        logging.info(f"Skipped: {results['skipped']}")
        logging.info(f"Errors: {results['errors']}")
        
        if results['errors'] == 0:
            logging.info("✅ All videos processed successfully!")
        else:
            logging.warning(f"⚠️ {results['errors']} videos had processing errors. Check the log for details.")
        
        logging.info("="*50)
        
    elif args.video_path:
        # Single video processing
        logging.info("="*50)
        logging.info("SINGLE VIDEO PROCESSING")
        logging.info("="*50)
        logging.info(f"Video path: {args.video_path}")
        logging.info("="*50)
        
        # Process single video
        output_path = processor.process_video(args.video_path, args.output)
        
        if output_path:
            logging.info("✅ MediaPipe landmark processing completed successfully!")
            logging.info(f"Output CSV: {output_path}")
        else:
            logging.error("❌ MediaPipe landmark processing failed!")
    else:
        logging.error("Please specify either --video_path for single video processing or --dataset_path for batch processing")
        parser.print_help()

if __name__ == "__main__":
    main() 