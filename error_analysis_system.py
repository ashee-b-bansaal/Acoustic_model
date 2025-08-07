# python error_analysis_system.py --experiment_dir /path/to/experiment --video_dataset /path/to/videoset --test_sessions 0901 --output_dir ./error_analysis --gpu_num 0 --custom_output_path /path/to/custom/output

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import json
import argparse
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our MediaPipe processors
from mediapipe_landmark_processor import MediaPipeLandmarkProcessor
from mediapipe_visualizer import MediaPipeVisualizer

parser = argparse.ArgumentParser(description='Comprehensive Error Analysis for Acoustic vs Vision-based Sign Language Classification')
parser.add_argument('--experiment_dir', default='./final_k5fold_testing', type=str, help='Path to experiment directory with existing results')
parser.add_argument('--video_dataset', default='', type=str, help='Path to video dataset folder')
parser.add_argument('--test_sessions', default='', type=str, help='Test sessions to analyze (optional for k-fold CV, used for video path resolution)')
parser.add_argument('--output_dir', default='./error_analysis', type=str, help='Output directory for analysis')
parser.add_argument('--create_visualizations', action='store_true', help='Create video visualizations for error cases')
parser.add_argument('--fold_number', default=None, type=int, help='Specific fold to analyze (if None, analyzes all folds)')
parser.add_argument('--gpu_num', default=0, type=int, help='GPU device number to use (default: 0)')
parser.add_argument('--custom_output_path', default='', type=str, help='Custom output directory path (overrides --output_dir)')

args = parser.parse_args()

# Configuration
experiment_dir = args.experiment_dir
video_dataset_path = args.video_dataset
test_sessions = args.test_sessions.split(',') if args.test_sessions else []
output_dir = args.custom_output_path if args.custom_output_path else args.output_dir
create_visualizations = args.create_visualizations
fold_number = args.fold_number
gpu_num = args.gpu_num

# Set GPU device
if gpu_num >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    print(f"Using GPU device: {gpu_num}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

def setup_logging():
    """Set up logging for error analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{output_dir}/error_analysis.log'),
            logging.StreamHandler()
        ]
    )

class ExperimentDataExtractor:
    """Extract data from existing experiment results."""
    
    def __init__(self, experiment_dir, fold_number=None):
        self.experiment_dir = experiment_dir
        self.fold_number = fold_number
        self.fold_dirs = self.find_fold_directories()
        
    def find_fold_directories(self):
        """Find all fold directories in the experiment directory."""
        fold_dirs = []
        if os.path.exists(self.experiment_dir):
            for item in os.listdir(self.experiment_dir):
                item_path = os.path.join(self.experiment_dir, item)
                if os.path.isdir(item_path) and 'fold_' in item:
                    fold_dirs.append(item_path)
        
        # Sort by fold number
        fold_dirs.sort(key=lambda x: int(x.split('fold_')[-1]))
        
        # Filter by specific fold if requested
        if self.fold_number is not None:
            fold_dirs = [d for d in fold_dirs if f'fold_{self.fold_number}' in d]
        
        return fold_dirs
    
    def extract_results_from_fold(self, fold_dir):
        """Extract results from a specific fold directory."""
        results = {
            'fold_dir': fold_dir,
            'fold_number': self.extract_fold_number(fold_dir),
            'test_results': None,
            'confusion_matrix': None,
            'log_file': None,
            'model_path': None
        }
        
        # Look for test results CSV
        test_results_path = os.path.join(fold_dir, 'test_results.csv')
        if os.path.exists(test_results_path):
            results['test_results'] = pd.read_csv(test_results_path)
            logging.info(f"Loaded test results from {test_results_path}: {len(results['test_results'])} samples")
        
        # Look for confusion matrix image
        cm_path = os.path.join(fold_dir, 'confusion_matrix.png')
        if os.path.exists(cm_path):
            results['confusion_matrix'] = cm_path
        
        # Look for log file
        log_path = os.path.join(fold_dir, 'logfile.txt')
        if os.path.exists(log_path):
            results['log_file'] = log_path
        
        # Look for model file
        model_path = os.path.join(fold_dir, 'best_model.pth')
        if os.path.exists(model_path):
            results['model_path'] = model_path
        
        return results
    
    def extract_fold_number(self, fold_dir):
        """Extract fold number from directory name."""
        try:
            return int(fold_dir.split('fold_')[-1])
        except:
            return 0
    
    def extract_all_results(self):
        """Extract results from all fold directories."""
        all_results = []
        
        for fold_dir in self.fold_dirs:
            results = self.extract_results_from_fold(fold_dir)
            all_results.append(results)
        
        logging.info(f"Extracted results from {len(all_results)} fold directories")
        return all_results
    
    def extract_error_cases(self, results):
        """Extract error cases from test results."""
        error_cases = []
        
        for fold_result in results:
            if fold_result['test_results'] is not None:
                df = fold_result['test_results']
                
                # Find error cases (where True Label != Predicted Label)
                error_df = df[df['True Label'] != df['Predicted Label']].copy()
                error_df['Fold'] = fold_result['fold_number']
                error_df['Fold_Dir'] = fold_result['fold_dir']
                
                error_cases.append(error_df)
        
        if error_cases:
            combined_errors = pd.concat(error_cases, ignore_index=True)
            logging.info(f"Found {len(combined_errors)} total error cases across all folds")
            return combined_errors
        else:
            logging.warning("No error cases found in test results")
            return pd.DataFrame()
    
    def extract_success_cases(self, results):
        """Extract success cases from test results."""
        success_cases = []
        
        for fold_result in results:
            if fold_result['test_results'] is not None:
                df = fold_result['test_results']
                
                # Find success cases (where True Label == Predicted Label)
                success_df = df[df['True Label'] == df['Predicted Label']].copy()
                success_df['Fold'] = fold_result['fold_number']
                success_df['Fold_Dir'] = fold_result['fold_dir']
                
                success_cases.append(success_df)
        
        if success_cases:
            combined_success = pd.concat(success_cases, ignore_index=True)
            logging.info(f"Found {len(combined_success)} total success cases across all folds")
            return combined_success
        else:
            logging.warning("No success cases found in test results")
            return pd.DataFrame()
    
    def extract_accuracy_stats(self, results):
        """Extract accuracy statistics from log files."""
        accuracy_stats = []
        
        for fold_result in results:
            if fold_result['log_file'] is not None:
                try:
                    with open(fold_result['log_file'], 'r') as f:
                        log_content = f.read()
                    
                    # Extract accuracy information
                    accuracy_info = {
                        'fold': fold_result['fold_number'],
                        'log_file': fold_result['log_file']
                    }
                    
                    # Look for accuracy patterns in log
                    import re
                    
                    # Find test accuracy
                    test_acc_matches = re.findall(r'Test Accuracy: (\d+\.\d+)%', log_content)
                    if test_acc_matches:
                        accuracy_info['test_accuracy'] = float(test_acc_matches[-1])
                    
                    # Find best accuracy
                    best_acc_matches = re.findall(r'Best model saved with Test Accuracy: (\d+\.\d+)%', log_content)
                    if best_acc_matches:
                        accuracy_info['best_accuracy'] = float(best_acc_matches[-1])
                    
                    # Find epoch information
                    epoch_matches = re.findall(r'Epoch \[(\d+)/(\d+)\]', log_content)
                    if epoch_matches:
                        accuracy_info['final_epoch'] = int(epoch_matches[-1][0])
                        accuracy_info['total_epochs'] = int(epoch_matches[-1][1])
                    
                    accuracy_stats.append(accuracy_info)
                    
                except Exception as e:
                    logging.error(f"Error reading log file {fold_result['log_file']}: {e}")
        
        return accuracy_stats

class ErrorAnalysisSystem:
    """Comprehensive error analysis system for acoustic vs vision-based classification."""
    
    def __init__(self, experiment_dir, video_dataset_path, output_dir):
        self.experiment_dir = experiment_dir
        self.video_dataset_path = video_dataset_path
        self.output_dir = output_dir
        self.processor = MediaPipeLandmarkProcessor()
        self.visualizer = MediaPipeVisualizer()
        
        # Extract data from existing experiments
        self.data_extractor = ExperimentDataExtractor(experiment_dir, fold_number)
        self.experiment_results = self.data_extractor.extract_all_results()
        
        # Extract error and success cases
        self.error_cases = self.data_extractor.extract_error_cases(self.experiment_results)
        self.success_cases = self.data_extractor.extract_success_cases(self.experiment_results)
        self.accuracy_stats = self.data_extractor.extract_accuracy_stats(self.experiment_results)
        
        # Initialize analysis containers
        self.landmark_analysis = {}
        self.temporal_analysis = {}
        
        logging.info("Error Analysis System initialized with existing experiment data")
    
    def load_acoustic_results(self):
        """Load acoustic model results."""
        try:
            df = pd.read_csv(self.acoustic_results_path)
            logging.info(f"Loaded acoustic results: {len(df)} samples")
            return df
        except Exception as e:
            logging.error(f"Error loading acoustic results: {e}")
            return None
    
    def extract_video_metadata(self, video_path):
        """Extract metadata from video file."""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'width': width,
                'height': height
            }
        except Exception as e:
            logging.error(f"Error extracting video metadata: {e}")
            return None
    
    def analyze_landmark_patterns(self, csv_path):
        """Analyze landmark patterns for error cases."""
        try:
            df = pd.read_csv(csv_path)
            
            # Extract key landmark groups
            pose_landmarks = []
            face_landmarks = []
            hand_landmarks = []
            
            # Pose landmarks (key points for sign language)
            pose_indices = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]  # Key pose points
            for idx in pose_indices:
                x = df[f'pose_{idx}_x'].fillna(0).values
                y = df[f'pose_{idx}_y'].fillna(0).values
                z = df[f'pose_{idx}_z'].fillna(0).values
                vis = df[f'pose_{idx}_visibility'].fillna(0).values
                pose_landmarks.append(np.column_stack([x, y, z, vis]))
            
            # Face landmarks (key facial features)
            face_indices = [10, 33, 133, 362, 263, 61, 291, 199, 419, 456, 356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
            for idx in face_indices:
                x = df[f'face_{idx}_x'].fillna(0).values
                y = df[f'face_{idx}_y'].fillna(0).values
                z = df[f'face_{idx}_z'].fillna(0).values
                face_landmarks.append(np.column_stack([x, y, z]))
            
            # Hand landmarks (crucial for sign language)
            for hand_idx in range(2):
                for i in range(21):
                    x = df[f'hand_{hand_idx}_{i}_x'].fillna(0).values
                    y = df[f'hand_{hand_idx}_{i}_y'].fillna(0).values
                    z = df[f'hand_{hand_idx}_{i}_z'].fillna(0).values
                    hand_landmarks.append(np.column_stack([x, y, z]))
            
            # Calculate motion metrics
            pose_motion = self.calculate_motion_metrics(pose_landmarks)
            face_motion = self.calculate_motion_metrics(face_landmarks)
            hand_motion = self.calculate_motion_metrics(hand_landmarks)
            
            # Calculate visibility metrics
            pose_visibility = self.calculate_visibility_metrics(df, 'pose')
            face_visibility = self.calculate_visibility_metrics(df, 'face')
            hand_visibility = self.calculate_visibility_metrics(df, 'hand')
            
            return {
                'pose_motion': pose_motion,
                'face_motion': face_motion,
                'hand_motion': hand_motion,
                'pose_visibility': pose_visibility,
                'face_visibility': face_visibility,
                'hand_visibility': hand_visibility,
                'frame_count': len(df)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing landmark patterns: {e}")
            return None
    
    def calculate_motion_metrics(self, landmarks):
        """Calculate motion metrics for landmark groups."""
        if not landmarks:
            return {}
        
        motion_metrics = {}
        
        # Calculate velocity (change in position over time)
        velocities = []
        for landmark in landmarks:
            if len(landmark) > 1:
                velocity = np.diff(landmark[:, :3], axis=0)  # Only x, y, z
                velocities.append(velocity)
        
        if velocities:
            all_velocities = np.concatenate(velocities, axis=0)
            motion_metrics['mean_velocity'] = np.mean(np.linalg.norm(all_velocities, axis=1))
            motion_metrics['max_velocity'] = np.max(np.linalg.norm(all_velocities, axis=1))
            motion_metrics['velocity_std'] = np.std(np.linalg.norm(all_velocities, axis=1))
        
        # Calculate acceleration
        accelerations = []
        for landmark in landmarks:
            if len(landmark) > 2:
                velocity = np.diff(landmark[:, :3], axis=0)
                acceleration = np.diff(velocity, axis=0)
                accelerations.append(acceleration)
        
        if accelerations:
            all_accelerations = np.concatenate(accelerations, axis=0)
            motion_metrics['mean_acceleration'] = np.mean(np.linalg.norm(all_accelerations, axis=1))
            motion_metrics['max_acceleration'] = np.max(np.linalg.norm(all_accelerations, axis=1))
        
        # Calculate motion complexity (entropy of motion)
        if velocities:
            motion_entropy = self.calculate_motion_entropy(velocities)
            motion_metrics['motion_entropy'] = motion_entropy
        
        return motion_metrics
    
    def calculate_motion_entropy(self, velocities):
        """Calculate entropy of motion patterns."""
        try:
            # Flatten all velocities
            all_velocities = np.concatenate(velocities, axis=0)
            
            # Discretize velocities into bins
            velocity_magnitudes = np.linalg.norm(all_velocities, axis=1)
            bins = np.histogram(velocity_magnitudes, bins=20)[0]
            
            # Calculate entropy
            bins = bins[bins > 0]  # Remove zero bins
            probabilities = bins / np.sum(bins)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            return entropy
        except:
            return 0.0
    
    def calculate_visibility_metrics(self, df, landmark_type):
        """Calculate visibility metrics for landmark types."""
        visibility_metrics = {}
        
        if landmark_type == 'pose':
            # Calculate average visibility for pose landmarks
            visibilities = []
            for i in range(33):
                vis_col = f'pose_{i}_visibility'
                if vis_col in df.columns:
                    vis = df[vis_col].fillna(0).values
                    visibilities.append(vis)
            
            if visibilities:
                all_vis = np.concatenate(visibilities)
                visibility_metrics['mean_visibility'] = np.mean(all_vis)
                visibility_metrics['min_visibility'] = np.min(all_vis)
                visibility_metrics['visibility_std'] = np.std(all_vis)
        
        elif landmark_type == 'face':
            # Face landmarks don't have visibility, use z-coordinate variation
            z_coords = []
            for i in range(478):
                z_col = f'face_{i}_z'
                if z_col in df.columns:
                    z = df[z_col].fillna(0).values
                    z_coords.append(z)
            
            if z_coords:
                all_z = np.concatenate(z_coords)
                visibility_metrics['z_variation'] = np.std(all_z)
                visibility_metrics['mean_z'] = np.mean(all_z)
        
        elif landmark_type == 'hand':
            # Hand landmarks don't have visibility, use detection confidence
            hand_detection = []
            for hand_idx in range(2):
                for i in range(21):
                    x_col = f'hand_{hand_idx}_{i}_x'
                    if x_col in df.columns:
                        x = df[x_col].fillna(0).values
                        # Count non-zero detections
                        detection_rate = np.sum(x != 0) / len(x)
                        hand_detection.append(detection_rate)
            
            if hand_detection:
                visibility_metrics['mean_detection_rate'] = np.mean(hand_detection)
                visibility_metrics['detection_std'] = np.std(hand_detection)
        
        return visibility_metrics
    
    def analyze_error_patterns(self):
        """Analyze patterns in error cases vs success cases from existing results."""
        logging.info("Starting error pattern analysis from existing results...")
        
        if self.error_cases.empty:
            logging.warning("No error cases found to analyze")
            return
        
        if self.success_cases.empty:
            logging.warning("No success cases found to analyze")
            return
        
        logging.info(f"Found {len(self.error_cases)} error cases and {len(self.success_cases)} success cases")
        
        # Generate comparative analysis
        self.generate_comparative_analysis()
        
        # Generate fold-wise analysis
        self.generate_fold_wise_analysis()
        
        # Generate label-wise analysis
        self.generate_label_wise_analysis()
        
        # Analyze video patterns for error cases (if video dataset provided)
        if self.video_dataset_path and os.path.exists(self.video_dataset_path):
            logging.info("Video dataset found - analyzing video patterns for error cases...")
            self.analyze_video_patterns_for_errors()
            
            # Compare success vs failure cases for specific labels
            self.compare_success_failure_patterns()
        else:
            logging.info("No video dataset provided - skipping video analysis")
    
    def find_video_path(self, video_name):
        """Find the actual video path for a given video name."""
        # If test_sessions are specified, use them
        if test_sessions:
            for session in test_sessions:
                session_path = os.path.join(self.video_dataset_path, session, "clips")
                if os.path.exists(session_path):
                    for file in os.listdir(session_path):
                        if video_name in file and file.endswith('.mp4'):
                            return os.path.join(session_path, file)
        
        # For k-fold CV, search all possible session directories
        if self.video_dataset_path and os.path.exists(self.video_dataset_path):
            for session_dir in os.listdir(self.video_dataset_path):
                session_path = os.path.join(self.video_dataset_path, session_dir, "clips")
                if os.path.exists(session_path):
                    for file in os.listdir(session_path):
                        if video_name in file and file.endswith('.mp4'):
                            return os.path.join(session_path, file)
        
        return None
    
    def analyze_single_case(self, row, video_path):
        """Analyze a single case (error or success)."""
        case_analysis = {
            'video_name': row['Video_Name'],
            'true_label': row['True Label'],
            'predicted_label': row['Predicted Label'],
            'confidence': row.get('Confidence', 0.0),
            'is_error': not row['Correct'],
            'video_metadata': self.extract_video_metadata(video_path)
        }
        
        # Process video to get landmarks
        csv_path = self.process_video_for_analysis(video_path)
        if csv_path:
            landmark_analysis = self.analyze_landmark_patterns(csv_path)
            case_analysis['landmark_analysis'] = landmark_analysis
        
        return case_analysis
    
    def process_video_for_analysis(self, video_path):
        """Process video to extract landmarks for analysis."""
        try:
            # Extract session and video info from path
            # Expected path: /path/to/videoset/session_0101/clips/sign_12_only_just.mp4
            path_parts = video_path.split(os.sep)
            
            # Find session directory and video name
            session_dir = None
            video_name = None
            
            for i, part in enumerate(path_parts):
                if part.startswith('session_'):
                    session_dir = part
                    # Video name should be in clips folder
                    if i + 2 < len(path_parts) and path_parts[i + 1] == 'clips':
                        video_name = path_parts[i + 2]
                        break
            
            if not session_dir or not video_name:
                logging.error(f"Could not parse session and video name from path: {video_path}")
                return None
            
            # Generate CSV path in dataset structure
            # Expected: /path/to/dataset/session_0101/csv/csv_12_only_just.csv
            dataset_path = self.video_dataset_path.replace('videoset', 'dataset')
            csv_name = f"csv_{video_name.replace('.mp4', '').replace('.avi', '').replace('.mov', '').replace('.mkv', '')}.csv"
            csv_path = os.path.join(dataset_path, session_dir, 'csv', csv_name)
            
            # Check if CSV already exists
            if not os.path.exists(csv_path):
                logging.info(f"Processing video for analysis: {video_path}")
                self.processor.process_video(video_path, csv_path)
            
            return csv_path if os.path.exists(csv_path) else None
            
        except Exception as e:
            logging.error(f"Error processing video for analysis: {e}")
            return None
    
    def generate_comparative_analysis(self):
        """Generate comparative analysis between error and success cases."""
        logging.info("Generating comparative analysis...")
        
        # Convert to DataFrames for easier analysis
        error_df = pd.DataFrame(self.error_cases)
        success_df = pd.DataFrame(self.success_cases)
        
        # Analyze motion patterns
        self.analyze_motion_patterns(error_df, success_df)
        
        # Analyze visibility patterns
        self.analyze_visibility_patterns(error_df, success_df)
        
        # Analyze temporal patterns
        self.analyze_temporal_patterns(error_df, success_df)
        
        # Generate visualizations
        self.generate_error_visualizations(error_df, success_df)
        
        # Save detailed reports
        self.save_analysis_reports(error_df, success_df)
    
    def generate_fold_wise_analysis(self):
        """Generate fold-wise analysis of errors."""
        logging.info("Generating fold-wise analysis...")
        
        if self.error_cases.empty:
            return
        
        # Analyze errors by fold
        fold_analysis = {}
        
        for fold_num in self.error_cases['Fold'].unique():
            fold_errors = self.error_cases[self.error_cases['Fold'] == fold_num]
            fold_success = self.success_cases[self.success_cases['Fold'] == fold_num]
            
            total_fold_cases = len(fold_errors) + len(fold_success)
            error_rate = len(fold_errors) / total_fold_cases * 100 if total_fold_cases > 0 else 0
            
            fold_analysis[fold_num] = {
                'error_count': len(fold_errors),
                'success_count': len(fold_success),
                'total_cases': total_fold_cases,
                'error_rate': error_rate,
                'most_common_errors': fold_errors['True Label'].value_counts().head(5).to_dict()
            }
        
        # Create fold-wise visualization
        self.plot_fold_wise_analysis(fold_analysis)
        
        # Save fold analysis
        with open(f'{self.output_dir}/fold_wise_analysis.json', 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {str(k): convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy scalar types
                    return obj.item()
                else:
                    return obj
            
            fold_analysis_serializable = convert_numpy_types(fold_analysis)
            json.dump(fold_analysis_serializable, f, indent=2)
        
        logging.info("Fold-wise analysis completed")
    
    def generate_label_wise_analysis(self):
        """Generate label-wise analysis of errors."""
        logging.info("Generating label-wise analysis...")
        
        if self.error_cases.empty:
            return
        
        # Analyze errors by true label
        label_analysis = {}
        
        for label in self.error_cases['True Label'].unique():
            label_errors = self.error_cases[self.error_cases['True Label'] == label]
            label_success = self.success_cases[self.success_cases['True Label'] == label]
            
            total_label_cases = len(label_errors) + len(label_success)
            error_rate = len(label_errors) / total_label_cases * 100 if total_label_cases > 0 else 0
            
            # Most common misclassifications for this label
            misclassifications = label_errors['Predicted Label'].value_counts().head(5).to_dict()
            
            label_analysis[label] = {
                'error_count': len(label_errors),
                'success_count': len(label_success),
                'total_cases': total_label_cases,
                'error_rate': error_rate,
                'most_common_misclassifications': misclassifications
            }
        
        # Create label-wise visualization
        self.plot_label_wise_analysis(label_analysis)
        
        # Save label analysis
        with open(f'{self.output_dir}/label_wise_analysis.json', 'w') as f:
            json.dump(label_analysis, f, indent=2)
        
        logging.info("Label-wise analysis completed")
    
    def plot_fold_wise_analysis(self, fold_analysis):
        """Plot fold-wise error analysis."""
        if not fold_analysis:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Error rate by fold
        folds = list(fold_analysis.keys())
        error_rates = [fold_analysis[f]['error_rate'] for f in folds]
        
        axes[0, 0].bar(folds, error_rates, color='red', alpha=0.7)
        axes[0, 0].set_title('Error Rate by Fold')
        axes[0, 0].set_xlabel('Fold Number')
        axes[0, 0].set_ylabel('Error Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error count by fold
        error_counts = [fold_analysis[f]['error_count'] for f in folds]
        success_counts = [fold_analysis[f]['success_count'] for f in folds]
        
        x = np.arange(len(folds))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, error_counts, width, label='Errors', color='red', alpha=0.7)
        axes[0, 1].bar(x + width/2, success_counts, width, label='Success', color='green', alpha=0.7)
        axes[0, 1].set_title('Error vs Success Count by Fold')
        axes[0, 1].set_xlabel('Fold Number')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(folds)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Total cases by fold
        total_cases = [fold_analysis[f]['total_cases'] for f in folds]
        axes[1, 0].bar(folds, total_cases, color='blue', alpha=0.7)
        axes[1, 0].set_title('Total Cases by Fold')
        axes[1, 0].set_xlabel('Fold Number')
        axes[1, 0].set_ylabel('Total Cases')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Average error rate
        avg_error_rate = np.mean(error_rates)
        axes[1, 1].axhline(y=avg_error_rate, color='red', linestyle='--', label=f'Average: {avg_error_rate:.1f}%')
        axes[1, 1].bar(folds, error_rates, color='red', alpha=0.7)
        axes[1, 1].set_title('Error Rate with Average')
        axes[1, 1].set_xlabel('Fold Number')
        axes[1, 1].set_ylabel('Error Rate (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fold_wise_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Fold-wise analysis plots saved")
    
    def plot_label_wise_analysis(self, label_analysis):
        """Plot label-wise error analysis."""
        if not label_analysis:
            return
        
        # Sort labels by error rate
        sorted_labels = sorted(label_analysis.items(), key=lambda x: x[1]['error_rate'], reverse=True)
        
        # Top 20 labels with highest error rates
        top_labels = sorted_labels[:20]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot 1: Error rate by label (top 20)
        labels = [item[0] for item in top_labels]
        error_rates = [item[1]['error_rate'] for item in top_labels]
        
        axes[0, 0].barh(labels, error_rates, color='red', alpha=0.7)
        axes[0, 0].set_title('Error Rate by Label (Top 20)')
        axes[0, 0].set_xlabel('Error Rate (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error count by label (top 20)
        error_counts = [item[1]['error_count'] for item in top_labels]
        
        axes[0, 1].barh(labels, error_counts, color='orange', alpha=0.7)
        axes[0, 1].set_title('Error Count by Label (Top 20)')
        axes[0, 1].set_xlabel('Error Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Total cases by label (top 20)
        total_cases = [item[1]['total_cases'] for item in top_labels]
        
        axes[1, 0].barh(labels, total_cases, color='blue', alpha=0.7)
        axes[1, 0].set_title('Total Cases by Label (Top 20)')
        axes[1, 0].set_xlabel('Total Cases')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Success vs Error ratio (top 20)
        success_counts = [item[1]['success_count'] for item in top_labels]
        
        x = np.arange(len(labels))
        width = 0.35
        
        axes[1, 1].barh(x - width/2, error_counts, width, label='Errors', color='red', alpha=0.7)
        axes[1, 1].barh(x + width/2, success_counts, width, label='Success', color='green', alpha=0.7)
        axes[1, 1].set_title('Success vs Error Count by Label (Top 20)')
        axes[1, 1].set_xlabel('Count')
        axes[1, 1].set_yticks(x)
        axes[1, 1].set_yticklabels(labels)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/label_wise_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Label-wise analysis plots saved")
    
    def analyze_video_patterns_for_errors(self):
        """Analyze video patterns specifically for error cases."""
        logging.info("Starting video pattern analysis for error cases...")
        
        # Create directory for video analysis results
        video_analysis_dir = f'{self.output_dir}/video_analysis'
        os.makedirs(video_analysis_dir, exist_ok=True)
        
        # Process only error cases for video analysis
        error_video_analysis = []
        
        for idx, error_case in self.error_cases.iterrows():
            try:
                video_name = error_case.get('Video_Name', '')
                if not video_name:
                    continue
                
                video_path = self.find_video_path(video_name)
                if video_path and os.path.exists(video_path):
                    logging.info(f"Analyzing video for error case: {video_name}")
                    
                    # Analyze video without re-running the model
                    video_analysis = self.analyze_single_video_case(error_case, video_path, video_analysis_dir)
                    if video_analysis:
                        error_video_analysis.append(video_analysis)
                else:
                    logging.warning(f"Video not found for error case: {video_name}")
                    
            except Exception as e:
                logging.error(f"Error analyzing video for case {idx}: {e}")
        
        # Generate video analysis summary
        if error_video_analysis:
            self.generate_video_analysis_summary(error_video_analysis, video_analysis_dir)
        
        logging.info(f"Video analysis completed for {len(error_video_analysis)} error cases")
    
    def analyze_single_video_case(self, error_case, video_path, output_dir):
        """Analyze a single video case without re-running the model."""
        try:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Extract video metadata
            video_metadata = self.extract_video_metadata(video_path)
            
            # Process video for landmarks (if not already processed)
            csv_path = self.process_video_for_analysis(video_path)
            
            video_analysis = {
                'video_name': video_name,
                'true_label': error_case['True Label'],
                'predicted_label': error_case['Predicted Label'],
                'fold': error_case.get('Fold', 'Unknown'),
                'video_metadata': video_metadata,
                'landmark_analysis': None,
                'motion_metrics': {},
                'visualization_path': None
            }
            
            # Analyze landmarks if CSV exists
            if csv_path and os.path.exists(csv_path):
                landmark_analysis = self.analyze_landmark_patterns(csv_path)
                video_analysis['landmark_analysis'] = landmark_analysis
                
                # Extract motion metrics
                if landmark_analysis:
                    video_analysis['motion_metrics'] = {
                        'hand_motion': landmark_analysis.get('hand_motion', {}),
                        'pose_motion': landmark_analysis.get('pose_motion', {}),
                        'face_motion': landmark_analysis.get('face_motion', {})
                    }
            
            # Create visualization if requested
            if create_visualizations and csv_path and os.path.exists(csv_path):
                viz_path = f"{output_dir}/{video_name}_error_analysis.mp4"
                try:
                    self.visualizer.create_landmark_visualization_from_csv(
                        video_path, csv_path, viz_path
                    )
                    video_analysis['visualization_path'] = viz_path
                    logging.info(f"Created visualization: {viz_path}")
                except Exception as e:
                    logging.error(f"Error creating visualization for {video_name}: {e}")
            
            return video_analysis
            
        except Exception as e:
            logging.error(f"Error analyzing video case {video_name}: {e}")
            return None
    
    def generate_video_analysis_summary(self, video_analyses, output_dir):
        """Generate summary of video analysis results."""
        logging.info("Generating video analysis summary...")
        
        # Extract motion metrics
        hand_motions = []
        pose_motions = []
        face_motions = []
        
        for analysis in video_analyses:
            if analysis['motion_metrics']:
                hand_motions.append(analysis['motion_metrics'].get('hand_motion', {}))
                pose_motions.append(analysis['motion_metrics'].get('pose_motion', {}))
                face_motions.append(analysis['motion_metrics'].get('face_motion', {}))
        
        # Calculate motion statistics
        motion_summary = {
            'total_error_videos_analyzed': len(video_analyses),
            'videos_with_motion_data': len(hand_motions),
            'hand_motion_stats': self.calculate_motion_statistics(hand_motions),
            'pose_motion_stats': self.calculate_motion_statistics(pose_motions),
            'face_motion_stats': self.calculate_motion_statistics(face_motions)
        }
        
        # Save motion summary
        with open(f'{output_dir}/motion_analysis_summary.json', 'w') as f:
            json.dump(motion_summary, f, indent=2)
        
        # Create motion visualization
        self.plot_error_case_motion_analysis(video_analyses, output_dir)
        
        # Generate detailed video analysis report
        self.generate_detailed_video_report(video_analyses, output_dir)
        
        logging.info("Video analysis summary generated")
    
    def calculate_motion_statistics(self, motion_data):
        """Calculate statistics from motion data."""
        if not motion_data:
            return {}
        
        # Extract velocity data
        velocities = []
        for motion in motion_data:
            if 'mean_velocity' in motion:
                velocities.append(motion['mean_velocity'])
        
        if velocities:
            return {
                'mean_velocity': np.mean(velocities),
                'std_velocity': np.std(velocities),
                'min_velocity': np.min(velocities),
                'max_velocity': np.max(velocities),
                'sample_count': len(velocities)
            }
        else:
            return {}
    
    def plot_error_case_motion_analysis(self, video_analyses, output_dir):
        """Plot motion analysis for error cases."""
        if not video_analyses:
            return
        
        # Extract motion data
        hand_velocities = []
        pose_velocities = []
        face_velocities = []
        labels = []
        
        for analysis in video_analyses:
            if analysis['motion_metrics']:
                hand_motion = analysis['motion_metrics'].get('hand_motion', {})
                pose_motion = analysis['motion_metrics'].get('pose_motion', {})
                face_motion = analysis['motion_metrics'].get('face_motion', {})
                
                hand_velocities.append(hand_motion.get('mean_velocity', 0))
                pose_velocities.append(pose_motion.get('mean_velocity', 0))
                face_velocities.append(face_motion.get('mean_velocity', 0))
                labels.append(f"{analysis['true_label']}→{analysis['predicted_label']}")
        
        if not hand_velocities:
            return
        
        # Create motion analysis plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Hand motion distribution
        axes[0, 0].hist(hand_velocities, bins=20, alpha=0.7, color='blue')
        axes[0, 0].set_title('Hand Motion Velocity Distribution (Error Cases)')
        axes[0, 0].set_xlabel('Mean Velocity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Pose motion distribution
        axes[0, 1].hist(pose_velocities, bins=20, alpha=0.7, color='green')
        axes[0, 1].set_title('Pose Motion Velocity Distribution (Error Cases)')
        axes[0, 1].set_xlabel('Mean Velocity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Face motion distribution
        axes[1, 0].hist(face_velocities, bins=20, alpha=0.7, color='red')
        axes[1, 0].set_title('Face Motion Velocity Distribution (Error Cases)')
        axes[1, 0].set_xlabel('Mean Velocity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Motion comparison
        motion_types = ['Hand', 'Pose', 'Face']
        motion_means = [np.mean(hand_velocities), np.mean(pose_velocities), np.mean(face_velocities)]
        
        axes[1, 1].bar(motion_types, motion_means, color=['blue', 'green', 'red'], alpha=0.7)
        axes[1, 1].set_title('Average Motion Velocity by Type (Error Cases)')
        axes[1, 1].set_ylabel('Mean Velocity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_case_motion_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Error case motion analysis plots saved")
    
    def generate_detailed_video_report(self, video_analyses, output_dir):
        """Generate detailed video analysis report."""
        logging.info("Generating detailed video analysis report...")
        
        # Create detailed report
        report_data = []
        
        for analysis in video_analyses:
            report_entry = {
                'video_name': analysis['video_name'],
                'true_label': analysis['true_label'],
                'predicted_label': analysis['predicted_label'],
                'fold': analysis['fold'],
                'video_duration': analysis['video_metadata'].get('duration', 'N/A'),
                'frame_count': analysis['video_metadata'].get('frame_count', 'N/A'),
                'hand_mean_velocity': analysis['motion_metrics'].get('hand_motion', {}).get('mean_velocity', 'N/A'),
                'pose_mean_velocity': analysis['motion_metrics'].get('pose_motion', {}).get('mean_velocity', 'N/A'),
                'face_mean_velocity': analysis['motion_metrics'].get('face_motion', {}).get('mean_velocity', 'N/A'),
                'visualization_path': analysis.get('visualization_path', 'N/A')
            }
            report_data.append(report_entry)
        
        # Save detailed report
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(f'{output_dir}/detailed_video_analysis.csv', index=False)
        
        # Generate markdown report
        with open(f'{output_dir}/video_analysis_report.md', 'w') as f:
            f.write("# Video Analysis Report for Error Cases\n\n")
            f.write(f"**Total Error Videos Analyzed:** {len(video_analyses)}\n\n")
            
            f.write("## Motion Analysis Summary\n\n")
            
            # Calculate motion statistics
            hand_velocities = [a['motion_metrics'].get('hand_motion', {}).get('mean_velocity', 0) 
                             for a in video_analyses if a['motion_metrics']]
            pose_velocities = [a['motion_metrics'].get('pose_motion', {}).get('mean_velocity', 0) 
                             for a in video_analyses if a['motion_metrics']]
            face_velocities = [a['motion_metrics'].get('face_motion', {}).get('mean_velocity', 0) 
                             for a in video_analyses if a['motion_metrics']]
            
            if hand_velocities:
                f.write(f"- **Hand Motion:** Mean = {np.mean(hand_velocities):.3f}, Std = {np.std(hand_velocities):.3f}\n")
            if pose_velocities:
                f.write(f"- **Pose Motion:** Mean = {np.mean(pose_velocities):.3f}, Std = {np.std(pose_velocities):.3f}\n")
            if face_velocities:
                f.write(f"- **Face Motion:** Mean = {np.mean(face_velocities):.3f}, Std = {np.std(face_velocities):.3f}\n")
            
            f.write("\n## Error Case Details\n\n")
            f.write("| Video | True Label | Predicted Label | Fold | Hand Motion | Pose Motion | Face Motion |\n")
            f.write("|-------|------------|-----------------|------|-------------|-------------|-------------|\n")
            
            for analysis in video_analyses:
                hand_vel = analysis['motion_metrics'].get('hand_motion', {}).get('mean_velocity', 'N/A')
                pose_vel = analysis['motion_metrics'].get('pose_motion', {}).get('mean_velocity', 'N/A')
                face_vel = analysis['motion_metrics'].get('face_motion', {}).get('mean_velocity', 'N/A')
                
                f.write(f"| {analysis['video_name']} | {analysis['true_label']} | {analysis['predicted_label']} | "
                       f"{analysis['fold']} | {hand_vel} | {pose_vel} | {face_vel} |\n")
        
        logging.info("Detailed video analysis report generated")
    
    def compare_success_failure_patterns(self):
        """Compare motion patterns between success and failure cases for specific labels."""
        logging.info("Starting success vs failure pattern comparison...")
        
        # Create directory for comparison results
        comparison_dir = f'{self.output_dir}/success_failure_comparison'
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Get all unique labels involved in errors
        error_labels = set()
        for _, error_case in self.error_cases.iterrows():
            error_labels.add(error_case['True Label'])
            error_labels.add(error_case['Predicted Label'])
        
        logging.info(f"Found {len(error_labels)} unique labels involved in errors")
        
        # Analyze each label
        label_comparisons = {}
        
        for label in error_labels:
            logging.info(f"Analyzing success vs failure patterns for label: {label}")
            
            # Get error cases where this label was the true label
            label_errors = self.error_cases[self.error_cases['True Label'] == label]
            
            # Get success cases where this label was correctly classified
            label_successes = self.success_cases[self.success_cases['True Label'] == label]
            
            if len(label_errors) > 0 and len(label_successes) > 0:
                comparison = self.compare_label_success_failure(label, label_errors, label_successes, comparison_dir)
                if comparison:
                    label_comparisons[label] = comparison
        
        # Generate overall comparison summary
        if label_comparisons:
            self.generate_comparison_summary(label_comparisons, comparison_dir)
        
        logging.info(f"Success vs failure comparison completed for {len(label_comparisons)} labels")
    
    def compare_label_success_failure(self, label, label_errors, label_successes, output_dir):
        """Compare success vs failure cases for a specific label."""
        try:
            # Process error cases for this label
            error_video_analyses = []
            for _, error_case in label_errors.iterrows():
                video_name = error_case.get('Video_Name', '')
                if video_name:
                    video_path = self.find_video_path(video_name)
                    if video_path and os.path.exists(video_path):
                        analysis = self.analyze_single_video_case(error_case, video_path, output_dir)
                        if analysis:
                            error_video_analyses.append(analysis)
            
            # Process success cases for this label
            success_video_analyses = []
            for _, success_case in label_successes.iterrows():
                video_name = success_case.get('Video_Name', '')
                if video_name:
                    video_path = self.find_video_path(video_name)
                    if video_path and os.path.exists(video_path):
                        analysis = self.analyze_single_video_case(success_case, video_path, output_dir)
                        if analysis:
                            success_video_analyses.append(analysis)
            
            # Compare motion patterns
            comparison = self.compare_motion_patterns(label, error_video_analyses, success_video_analyses, output_dir)
            
            return comparison
            
        except Exception as e:
            logging.error(f"Error comparing success/failure for label {label}: {e}")
            return None
    
    def compare_motion_patterns(self, label, error_analyses, success_analyses, output_dir):
        """Compare motion patterns between error and success cases for a label."""
        if not error_analyses or not success_analyses:
            return None
        
        # Extract motion metrics
        error_hand_velocities = []
        error_pose_velocities = []
        error_face_velocities = []
        
        success_hand_velocities = []
        success_pose_velocities = []
        success_face_velocities = []
        
        # Extract error motion data
        for analysis in error_analyses:
            if analysis['motion_metrics']:
                hand_motion = analysis['motion_metrics'].get('hand_motion', {})
                pose_motion = analysis['motion_metrics'].get('pose_motion', {})
                face_motion = analysis['motion_metrics'].get('face_motion', {})
                
                error_hand_velocities.append(hand_motion.get('mean_velocity', 0))
                error_pose_velocities.append(pose_motion.get('mean_velocity', 0))
                error_face_velocities.append(face_motion.get('mean_velocity', 0))
        
        # Extract success motion data
        for analysis in success_analyses:
            if analysis['motion_metrics']:
                hand_motion = analysis['motion_metrics'].get('hand_motion', {})
                pose_motion = analysis['motion_metrics'].get('pose_motion', {})
                face_motion = analysis['motion_metrics'].get('face_motion', {})
                
                success_hand_velocities.append(hand_motion.get('mean_velocity', 0))
                success_pose_velocities.append(pose_motion.get('mean_velocity', 0))
                success_face_velocities.append(face_motion.get('mean_velocity', 0))
        
        # Calculate statistics
        comparison = {
            'label': label,
            'error_count': len(error_analyses),
            'success_count': len(success_analyses),
            'hand_motion': {
                'error_mean': np.mean(error_hand_velocities) if error_hand_velocities else 0,
                'error_std': np.std(error_hand_velocities) if error_hand_velocities else 0,
                'success_mean': np.mean(success_hand_velocities) if success_hand_velocities else 0,
                'success_std': np.std(success_hand_velocities) if success_hand_velocities else 0,
                'difference': (np.mean(error_hand_velocities) if error_hand_velocities else 0) - 
                            (np.mean(success_hand_velocities) if success_hand_velocities else 0)
            },
            'pose_motion': {
                'error_mean': np.mean(error_pose_velocities) if error_pose_velocities else 0,
                'error_std': np.std(error_pose_velocities) if error_pose_velocities else 0,
                'success_mean': np.mean(success_pose_velocities) if success_pose_velocities else 0,
                'success_std': np.std(success_pose_velocities) if success_pose_velocities else 0,
                'difference': (np.mean(error_pose_velocities) if error_pose_velocities else 0) - 
                            (np.mean(success_pose_velocities) if success_pose_velocities else 0)
            },
            'face_motion': {
                'error_mean': np.mean(error_face_velocities) if error_face_velocities else 0,
                'error_std': np.std(error_face_velocities) if error_face_velocities else 0,
                'success_mean': np.mean(success_face_velocities) if success_face_velocities else 0,
                'success_std': np.std(success_face_velocities) if success_face_velocities else 0,
                'difference': (np.mean(error_face_velocities) if error_face_velocities else 0) - 
                            (np.mean(success_face_velocities) if success_face_velocities else 0)
            }
        }
        
        # Create visualization for this label
        self.plot_label_comparison(label, comparison, error_hand_velocities, success_hand_velocities,
                                 error_pose_velocities, success_pose_velocities,
                                 error_face_velocities, success_face_velocities, output_dir)
        
        return comparison
    
    def plot_label_comparison(self, label, comparison, error_hand, success_hand, 
                            error_pose, success_pose, error_face, success_face, output_dir):
        """Create comparison plots for a specific label."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Hand motion comparison
        if error_hand and success_hand:
            axes[0, 0].hist(error_hand, bins=15, alpha=0.7, label='Error Cases', color='red')
            axes[0, 0].hist(success_hand, bins=15, alpha=0.7, label='Success Cases', color='green')
            axes[0, 0].set_title(f'Hand Motion Comparison - {label}')
            axes[0, 0].set_xlabel('Mean Velocity')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Pose motion comparison
        if error_pose and success_pose:
            axes[0, 1].hist(error_pose, bins=15, alpha=0.7, label='Error Cases', color='red')
            axes[0, 1].hist(success_pose, bins=15, alpha=0.7, label='Success Cases', color='green')
            axes[0, 1].set_title(f'Pose Motion Comparison - {label}')
            axes[0, 1].set_xlabel('Mean Velocity')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Face motion comparison
        if error_face and success_face:
            axes[1, 0].hist(error_face, bins=15, alpha=0.7, label='Error Cases', color='red')
            axes[1, 0].hist(success_face, bins=15, alpha=0.7, label='Success Cases', color='green')
            axes[1, 0].set_title(f'Face Motion Comparison - {label}')
            axes[1, 0].set_xlabel('Mean Velocity')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Motion difference summary
        motion_types = ['Hand', 'Pose', 'Face']
        differences = [
            comparison['hand_motion']['difference'],
            comparison['pose_motion']['difference'],
            comparison['face_motion']['difference']
        ]
        
        colors = ['red' if d > 0 else 'blue' for d in differences]
        axes[1, 1].bar(motion_types, differences, color=colors, alpha=0.7)
        axes[1, 1].set_title(f'Motion Difference (Error - Success) - {label}')
        axes[1, 1].set_ylabel('Velocity Difference')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{label}_success_failure_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Comparison plots saved for label: {label}")
    
    def generate_comparison_summary(self, label_comparisons, output_dir):
        """Generate summary of all label comparisons."""
        logging.info("Generating comparison summary...")
        
        # Create summary report
        summary_data = []
        
        for label, comparison in label_comparisons.items():
            summary_entry = {
                'label': label,
                'error_count': comparison['error_count'],
                'success_count': comparison['success_count'],
                'hand_motion_error_mean': comparison['hand_motion']['error_mean'],
                'hand_motion_success_mean': comparison['hand_motion']['success_mean'],
                'hand_motion_difference': comparison['hand_motion']['difference'],
                'pose_motion_error_mean': comparison['pose_motion']['error_mean'],
                'pose_motion_success_mean': comparison['pose_motion']['success_mean'],
                'pose_motion_difference': comparison['pose_motion']['difference'],
                'face_motion_error_mean': comparison['face_motion']['error_mean'],
                'face_motion_success_mean': comparison['face_motion']['success_mean'],
                'face_motion_difference': comparison['face_motion']['difference']
            }
            summary_data.append(summary_entry)
        
        # Save summary CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/label_comparison_summary.csv', index=False)
        
        # Generate markdown report
        with open(f'{output_dir}/success_failure_comparison_report.md', 'w') as f:
            f.write("# Success vs Failure Motion Pattern Comparison\n\n")
            f.write(f"**Total Labels Analyzed:** {len(label_comparisons)}\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Find labels with significant differences
            significant_labels = []
            for label, comp in label_comparisons.items():
                hand_diff = abs(comp['hand_motion']['difference'])
                pose_diff = abs(comp['pose_motion']['difference'])
                face_diff = abs(comp['face_motion']['difference'])
                
                if hand_diff > 0.1 or pose_diff > 0.1 or face_diff > 0.1:
                    significant_labels.append((label, hand_diff + pose_diff + face_diff))
            
            if significant_labels:
                significant_labels.sort(key=lambda x: x[1], reverse=True)
                f.write("### Labels with Significant Motion Differences\n\n")
                for label, total_diff in significant_labels[:10]:
                    comp = label_comparisons[label]
                    f.write(f"#### {label}\n")
                    f.write(f"- **Hand Motion:** Error={comp['hand_motion']['error_mean']:.3f}, "
                           f"Success={comp['hand_motion']['success_mean']:.3f}, "
                           f"Diff={comp['hand_motion']['difference']:.3f}\n")
                    f.write(f"- **Pose Motion:** Error={comp['pose_motion']['error_mean']:.3f}, "
                           f"Success={comp['pose_motion']['success_mean']:.3f}, "
                           f"Diff={comp['pose_motion']['difference']:.3f}\n")
                    f.write(f"- **Face Motion:** Error={comp['face_motion']['error_mean']:.3f}, "
                           f"Success={comp['face_motion']['success_mean']:.3f}, "
                           f"Diff={comp['face_motion']['difference']:.3f}\n\n")
            
            f.write("## Detailed Comparison Table\n\n")
            f.write("| Label | Error Count | Success Count | Hand Diff | Pose Diff | Face Diff |\n")
            f.write("|-------|-------------|---------------|-----------|-----------|-----------|\n")
            
            for label, comp in label_comparisons.items():
                f.write(f"| {label} | {comp['error_count']} | {comp['success_count']} | "
                       f"{comp['hand_motion']['difference']:.3f} | "
                       f"{comp['pose_motion']['difference']:.3f} | "
                       f"{comp['face_motion']['difference']:.3f} |\n")
        
        # Save detailed comparison data
        with open(f'{output_dir}/detailed_comparison_data.json', 'w') as f:
            json.dump(label_comparisons, f, indent=2)
        
        logging.info("Comparison summary generated")
    
    def analyze_motion_patterns(self, error_df, success_df):
        """Analyze motion patterns in error vs success cases."""
        logging.info("Analyzing motion patterns...")
        
        # Extract motion metrics
        error_motions = []
        success_motions = []
        
        for case in self.error_cases:
            if 'landmark_analysis' in case and case['landmark_analysis']:
                error_motions.append(case['landmark_analysis'])
        
        for case in self.success_cases:
            if 'landmark_analysis' in case and case['landmark_analysis']:
                success_motions.append(case['landmark_analysis'])
        
        # Compare motion metrics
        motion_comparison = {}
        
        for metric in ['pose_motion', 'face_motion', 'hand_motion']:
            error_metrics = [m[metric] for m in error_motions if metric in m]
            success_metrics = [m[metric] for m in success_motions if metric in m]
            
            if error_metrics and success_metrics:
                motion_comparison[metric] = {
                    'error_mean': np.mean([m.get('mean_velocity', 0) for m in error_metrics]),
                    'success_mean': np.mean([m.get('mean_velocity', 0) for m in success_metrics]),
                    'error_std': np.std([m.get('mean_velocity', 0) for m in error_metrics]),
                    'success_std': np.std([m.get('mean_velocity', 0) for m in success_metrics])
                }
        
        self.motion_analysis = motion_comparison
        logging.info(f"Motion analysis completed: {len(motion_comparison)} metrics compared")
    
    def analyze_visibility_patterns(self, error_df, success_df):
        """Analyze visibility patterns in error vs success cases."""
        logging.info("Analyzing visibility patterns...")
        
        # Extract visibility metrics
        error_visibilities = []
        success_visibilities = []
        
        for case in self.error_cases:
            if 'landmark_analysis' in case and case['landmark_analysis']:
                error_visibilities.append(case['landmark_analysis'])
        
        for case in self.success_cases:
            if 'landmark_analysis' in case and case['landmark_analysis']:
                success_visibilities.append(case['landmark_analysis'])
        
        # Compare visibility metrics
        visibility_comparison = {}
        
        for metric in ['pose_visibility', 'face_visibility', 'hand_visibility']:
            error_metrics = [v[metric] for v in error_visibilities if metric in v]
            success_metrics = [v[metric] for v in success_visibilities if metric in v]
            
            if error_metrics and success_metrics:
                visibility_comparison[metric] = {
                    'error_mean': np.mean([m.get('mean_visibility', 0) for m in error_metrics]),
                    'success_mean': np.mean([m.get('mean_visibility', 0) for m in success_metrics]),
                    'error_std': np.std([m.get('mean_visibility', 0) for m in error_metrics]),
                    'success_std': np.std([m.get('mean_visibility', 0) for m in success_metrics])
                }
        
        self.visibility_analysis = visibility_comparison
        logging.info(f"Visibility analysis completed: {len(visibility_comparison)} metrics compared")
    
    def analyze_temporal_patterns(self, error_df, success_df):
        """Analyze temporal patterns in error vs success cases."""
        logging.info("Analyzing temporal patterns...")
        
        # Extract temporal metrics
        error_temporal = []
        success_temporal = []
        
        for case in self.error_cases:
            if 'landmark_analysis' in case and case['landmark_analysis']:
                error_temporal.append(case['landmark_analysis'])
        
        for case in self.success_cases:
            if 'landmark_analysis' in case and case['landmark_analysis']:
                success_temporal.append(case['landmark_analysis'])
        
        # Compare temporal metrics
        temporal_comparison = {
            'frame_count': {
                'error_mean': np.mean([t.get('frame_count', 0) for t in error_temporal]),
                'success_mean': np.mean([t.get('frame_count', 0) for t in success_temporal]),
                'error_std': np.std([t.get('frame_count', 0) for t in error_temporal]),
                'success_std': np.std([t.get('frame_count', 0) for t in success_temporal])
            }
        }
        
        self.temporal_analysis = temporal_comparison
        logging.info("Temporal analysis completed")
    
    def generate_error_visualizations(self, error_df, success_df):
        """Generate visualizations for error analysis."""
        logging.info("Generating error visualizations...")
        
        # 1. Motion comparison plot
        self.plot_motion_comparison()
        
        # 2. Visibility comparison plot
        self.plot_visibility_comparison()
        
        # 3. Error case distribution
        self.plot_error_distribution()
        
        # 4. Confidence vs motion correlation
        self.plot_confidence_motion_correlation()
        
        # 5. Create video visualizations for error cases
        if create_visualizations:
            self.create_error_case_videos()
    
    def plot_motion_comparison(self):
        """Plot motion comparison between error and success cases."""
        if not hasattr(self, 'motion_analysis'):
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['pose_motion', 'face_motion', 'hand_motion']
        titles = ['Pose Motion', 'Face Motion', 'Hand Motion']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if metric in self.motion_analysis:
                data = self.motion_analysis[metric]
                
                # Create comparison bars
                categories = ['Error Cases', 'Success Cases']
                means = [data['error_mean'], data['success_mean']]
                stds = [data['error_std'], data['success_std']]
                
                bars = axes[i].bar(categories, means, yerr=stds, capsize=5)
                axes[i].set_title(f'{title} Comparison')
                axes[i].set_ylabel('Mean Velocity')
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, mean in zip(bars, means):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/motion_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Motion comparison plot saved")
    
    def plot_visibility_comparison(self):
        """Plot visibility comparison between error and success cases."""
        if not hasattr(self, 'visibility_analysis'):
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['pose_visibility', 'face_visibility', 'hand_visibility']
        titles = ['Pose Visibility', 'Face Visibility', 'Hand Detection']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            if metric in self.visibility_analysis:
                data = self.visibility_analysis[metric]
                
                # Create comparison bars
                categories = ['Error Cases', 'Success Cases']
                means = [data['error_mean'], data['success_mean']]
                stds = [data['error_std'], data['success_std']]
                
                bars = axes[i].bar(categories, means, yerr=stds, capsize=5)
                axes[i].set_title(f'{title} Comparison')
                axes[i].set_ylabel('Visibility/Detection Rate')
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, mean in zip(bars, means):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/visibility_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("Visibility comparison plot saved")
    
    def plot_error_distribution(self):
        """Plot error case distribution."""
        if self.error_cases.empty:
            return
        
        # Analyze error patterns
        error_labels = self.error_cases['True Label'].tolist()
        predicted_labels = self.error_cases['Predicted Label'].tolist()
        
        # Create confusion matrix for errors
        cm = confusion_matrix(error_labels, predicted_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
        plt.title('Error Case Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/error_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot error distribution by true label
        label_counts = pd.Series(error_labels).value_counts()
        plt.figure(figsize=(12, 6))
        label_counts.plot(kind='bar')
        plt.title('Error Distribution by True Label')
        plt.xlabel('True Label')
        plt.ylabel('Number of Errors')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Error distribution plots saved")
    
    def plot_confidence_motion_correlation(self):
        """Plot correlation between confidence and motion metrics."""
        if self.error_cases.empty:
            return
        
        # Extract confidence and motion data
        confidences = []
        motion_velocities = []
        
        for idx, case in self.error_cases.iterrows():
            if 'confidence' in case and 'landmark_analysis' in case:
                confidences.append(case['confidence'])
                if case['landmark_analysis'] and 'hand_motion' in case['landmark_analysis']:
                    motion_velocities.append(case['landmark_analysis']['hand_motion'].get('mean_velocity', 0))
                else:
                    motion_velocities.append(0)
        
        if confidences and motion_velocities:
            plt.figure(figsize=(10, 6))
            plt.scatter(confidences, motion_velocities, alpha=0.6)
            plt.xlabel('Model Confidence')
            plt.ylabel('Hand Motion Velocity')
            plt.title('Confidence vs Hand Motion Correlation (Error Cases)')
            
            # Add trend line
            z = np.polyfit(confidences, motion_velocities, 1)
            p = np.poly1d(z)
            plt.plot(confidences, p(confidences), "r--", alpha=0.8)
            
            # Calculate correlation
            correlation = np.corrcoef(confidences, motion_velocities)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor="white"))
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/confidence_motion_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Confidence-motion correlation plot saved (r={correlation:.3f})")
    
    def create_error_case_videos(self):
        """Create video visualizations for error cases."""
        logging.info("Creating error case video visualizations...")
        
        # Check if we have video information
        if not hasattr(self, 'video_dataset_path') or not self.video_dataset_path:
            logging.warning("No video dataset path provided. Skipping video creation.")
            return
            
        # Check if we have video name column
        video_name_columns = ['Video_Name', 'video_name', 'filename', 'file_name', 'video_path']
        available_video_column = None
        for col in video_name_columns:
            if col in self.error_cases.columns:
                available_video_column = col
                break
        
        if not available_video_column:
            logging.warning("No video name column found in test results. Video creation requires video file information.")
            logging.info("Available columns: " + ", ".join(self.error_cases.columns))
            logging.info("To enable video creation, ensure test_results.csv includes video file names.")
            return
        
        error_videos_dir = f'{self.output_dir}/error_videos'
        os.makedirs(error_videos_dir, exist_ok=True)
        
        for i, (idx, case) in enumerate(self.error_cases.iterrows()):
            try:
                video_name = case[available_video_column]
                video_path = self.find_video_path(video_name)
                
                if video_path:
                    # Create recreated video with landmarks
                    csv_path = f"csv_data/{video_name}_mediapipe_landmarks.csv"
                    output_video_path = f"{error_videos_dir}/error_case_{i}_{video_name}_analysis.mp4"
                    
                    if os.path.exists(csv_path):
                        self.visualizer.create_landmark_visualization_from_csv(
                            video_path, csv_path, output_video_path
                        )
                        logging.info(f"Created error case video: {output_video_path}")
                
            except Exception as e:
                logging.error(f"Error creating video for case {i}: {e}")
    
    def save_analysis_reports(self, error_df, success_df):
        """Save detailed analysis reports."""
        logging.info("Saving analysis reports...")
        
        # 1. Summary statistics
        total_cases = len(self.error_cases) + len(self.success_cases)
        error_rate = len(self.error_cases) / total_cases * 100 if total_cases > 0 else 0
        success_rate = len(self.success_cases) / total_cases * 100 if total_cases > 0 else 0
        
        summary_stats = {
            'total_cases': total_cases,
            'error_cases': len(self.error_cases),
            'success_cases': len(self.success_cases),
            'error_rate': error_rate,
            'success_rate': success_rate,
            'accuracy_stats': self.accuracy_stats
        }
        
        with open(f'{self.output_dir}/summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # 2. Save error and success cases
        if not self.error_cases.empty:
            self.error_cases.to_csv(f'{self.output_dir}/error_cases.csv', index=False)
        
        if not self.success_cases.empty:
            self.success_cases.to_csv(f'{self.output_dir}/success_cases.csv', index=False)
        
        # 3. Comparative analysis report
        comparison_report = {
            'motion_analysis': getattr(self, 'motion_analysis', {}),
            'visibility_analysis': getattr(self, 'visibility_analysis', {}),
            'temporal_analysis': getattr(self, 'temporal_analysis', {})
        }
        
        with open(f'{self.output_dir}/comparative_analysis.json', 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        # 4. Key findings report
        self.generate_key_findings_report()
        
        # 5. Experiment summary
        self.generate_experiment_summary()
        
        logging.info("Analysis reports saved")
    
    def generate_experiment_summary(self):
        """Generate a summary of the experiment results."""
        logging.info("Generating experiment summary...")
        
        summary = {
            'experiment_directory': self.experiment_dir,
            'total_folds_analyzed': len(self.experiment_results),
            'fold_directories': [r['fold_dir'] for r in self.experiment_results],
            'accuracy_by_fold': {}
        }
        
        # Add accuracy by fold
        for stat in self.accuracy_stats:
            fold_num = stat['fold']
            summary['accuracy_by_fold'][fold_num] = {
                'test_accuracy': stat.get('test_accuracy', 'N/A'),
                'best_accuracy': stat.get('best_accuracy', 'N/A'),
                'final_epoch': stat.get('final_epoch', 'N/A'),
                'total_epochs': stat.get('total_epochs', 'N/A')
            }
        
        # Calculate overall statistics
        if self.accuracy_stats:
            accuracies = [s.get('best_accuracy', 0) for s in self.accuracy_stats if s.get('best_accuracy')]
            if accuracies:
                summary['overall_statistics'] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies)
                }
        
        # Save experiment summary
        with open(f'{self.output_dir}/experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate markdown summary
        with open(f'{self.output_dir}/experiment_summary.md', 'w') as f:
            f.write("# Experiment Summary\n\n")
            f.write(f"**Experiment Directory:** {self.experiment_dir}\n\n")
            f.write(f"**Total Folds Analyzed:** {len(self.experiment_results)}\n\n")
            
            f.write("## Accuracy by Fold\n\n")
            for fold_num, acc_info in summary['accuracy_by_fold'].items():
                f.write(f"### Fold {fold_num}\n")
                f.write(f"- Test Accuracy: {acc_info['test_accuracy']}%\n")
                f.write(f"- Best Accuracy: {acc_info['best_accuracy']}%\n")
                f.write(f"- Final Epoch: {acc_info['final_epoch']}/{acc_info['total_epochs']}\n\n")
            
            if 'overall_statistics' in summary:
                overall = summary['overall_statistics']
                f.write("## Overall Statistics\n\n")
                f.write(f"- Mean Accuracy: {overall['mean_accuracy']:.2f}%\n")
                f.write(f"- Standard Deviation: {overall['std_accuracy']:.2f}%\n")
                f.write(f"- Min Accuracy: {overall['min_accuracy']:.2f}%\n")
                f.write(f"- Max Accuracy: {overall['max_accuracy']:.2f}%\n\n")
            
            f.write("## Error Analysis Summary\n\n")
            f.write(f"- Total Cases: {len(self.error_cases) + len(self.success_cases)}\n")
            f.write(f"- Error Cases: {len(self.error_cases)} ({len(self.error_cases)/(len(self.error_cases) + len(self.success_cases))*100:.1f}%)\n")
            f.write(f"- Success Cases: {len(self.success_cases)} ({len(self.success_cases)/(len(self.error_cases) + len(self.success_cases))*100:.1f}%)\n")
        
        logging.info("Experiment summary generated")
    
    def generate_key_findings_report(self):
        """Generate a key findings report for the research paper."""
        findings = []
        
        # Analyze motion differences
        if hasattr(self, 'motion_analysis'):
            for metric, data in self.motion_analysis.items():
                error_mean = data['error_mean']
                success_mean = data['success_mean']
                difference = abs(error_mean - success_mean)
                percentage_diff = (difference / success_mean) * 100 if success_mean > 0 else 0
                
                findings.append({
                    'metric': f'{metric}_velocity',
                    'error_mean': error_mean,
                    'success_mean': success_mean,
                    'difference': difference,
                    'percentage_difference': percentage_diff,
                    'finding': f"Error cases show {percentage_diff:.1f}% {'higher' if error_mean > success_mean else 'lower'} {metric} motion than success cases"
                })
        
        # Analyze visibility differences
        if hasattr(self, 'visibility_analysis'):
            for metric, data in self.visibility_analysis.items():
                error_mean = data['error_mean']
                success_mean = data['success_mean']
                difference = abs(error_mean - success_mean)
                percentage_diff = (difference / success_mean) * 100 if success_mean > 0 else 0
                
                findings.append({
                    'metric': f'{metric}_visibility',
                    'error_mean': error_mean,
                    'success_mean': success_mean,
                    'difference': difference,
                    'percentage_difference': percentage_diff,
                    'finding': f"Error cases show {percentage_diff:.1f}% {'higher' if error_mean > success_mean else 'lower'} {metric} visibility than success cases"
                })
        
        # Save findings
        findings_df = pd.DataFrame(findings)
        findings_df.to_csv(f'{self.output_dir}/key_findings.csv', index=False)
        
        # Generate markdown report
        with open(f'{self.output_dir}/key_findings_report.md', 'w') as f:
            f.write("# Error Analysis Key Findings\n\n")
            f.write(f"## Summary\n")
            f.write(f"- Total cases analyzed: {len(self.error_cases) + len(self.success_cases)}\n")
            f.write(f"- Error cases: {len(self.error_cases)} ({len(self.error_cases)/(len(self.error_cases) + len(self.success_cases))*100:.1f}%)\n")
            f.write(f"- Success cases: {len(self.success_cases)} ({len(self.success_cases)/(len(self.error_cases) + len(self.success_cases))*100:.1f}%)\n\n")
            
            f.write("## Key Findings\n\n")
            for finding in findings:
                f.write(f"### {finding['metric'].replace('_', ' ').title()}\n")
                f.write(f"- {finding['finding']}\n")
                f.write(f"- Error mean: {finding['error_mean']:.3f}\n")
                f.write(f"- Success mean: {finding['success_mean']:.3f}\n")
                f.write(f"- Difference: {finding['difference']:.3f} ({finding['percentage_difference']:.1f}%)\n\n")
        
        logging.info("Key findings report generated")

def main():
    """Main execution function."""
    setup_logging()
    
    logging.info("="*50)
    logging.info("Starting Comprehensive Error Analysis from Existing Results")
    logging.info("="*50)
    
    # Initialize error analysis system
    analyzer = ErrorAnalysisSystem(experiment_dir, video_dataset_path, output_dir)
    
    # Run comprehensive analysis
    analyzer.analyze_error_patterns()
    
    logging.info("="*50)
    logging.info("Error Analysis Completed Successfully!")
    logging.info(f"Results saved in: {output_dir}")
    logging.info("="*50)

if __name__ == "__main__":
    main() 