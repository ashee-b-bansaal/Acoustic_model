# python error_analysis_system.py --acoustic_results /path/to/acoustic_results.csv --video_dataset /path/to/videoset --test_sessions 0901 --output_dir ./error_analysis

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
parser.add_argument('--acoustic_results', default='', type=str, help='Path to acoustic model results CSV')
parser.add_argument('--video_dataset', default='', type=str, help='Path to video dataset folder')
parser.add_argument('--test_sessions', default='', type=str, help='Test sessions to analyze')
parser.add_argument('--output_dir', default='./error_analysis', type=str, help='Output directory for analysis')
parser.add_argument('--confidence_threshold', default=0.8, type=float, help='Confidence threshold for error analysis')
parser.add_argument('--create_visualizations', action='store_true', help='Create video visualizations for error cases')

args = parser.parse_args()

# Configuration
acoustic_results_path = args.acoustic_results
video_dataset_path = args.video_dataset
test_sessions = args.test_sessions.split(',') if args.test_sessions else []
output_dir = args.output_dir
confidence_threshold = args.confidence_threshold
create_visualizations = args.create_visualizations

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

class ErrorAnalysisSystem:
    """Comprehensive error analysis system for acoustic vs vision-based classification."""
    
    def __init__(self, acoustic_results_path, video_dataset_path, output_dir):
        self.acoustic_results_path = acoustic_results_path
        self.video_dataset_path = video_dataset_path
        self.output_dir = output_dir
        self.processor = MediaPipeLandmarkProcessor()
        self.visualizer = MediaPipeVisualizer()
        
        # Load acoustic results
        self.acoustic_results = self.load_acoustic_results()
        
        # Initialize analysis containers
        self.error_cases = []
        self.success_cases = []
        self.landmark_analysis = {}
        self.temporal_analysis = {}
        
        logging.info("Error Analysis System initialized")
    
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
        """Analyze patterns in error cases vs success cases."""
        logging.info("Starting error pattern analysis...")
        
        # Separate error and success cases
        error_cases = self.acoustic_results[self.acoustic_results['Correct'] == False]
        success_cases = self.acoustic_results[self.acoustic_results['Correct'] == True]
        
        logging.info(f"Found {len(error_cases)} error cases and {len(success_cases)} success cases")
        
        # Analyze each case
        for idx, row in self.acoustic_results.iterrows():
            video_path = self.find_video_path(row['Video_Name'])
            if video_path:
                analysis = self.analyze_single_case(row, video_path)
                if row['Correct'] == False:
                    self.error_cases.append(analysis)
                else:
                    self.success_cases.append(analysis)
        
        # Generate comparative analysis
        self.generate_comparative_analysis()
    
    def find_video_path(self, video_name):
        """Find the actual video path for a given video name."""
        for session in test_sessions:
            session_path = os.path.join(self.video_dataset_path, session, "clips")
            if os.path.exists(session_path):
                for file in os.listdir(session_path):
                    if video_name in file and file.endswith('.mp4'):
                        return os.path.join(session_path, file)
        return None
    
    def analyze_single_case(self, row, video_path):
        """Analyze a single case (error or success)."""
        case_analysis = {
            'video_name': row['Video_Name'],
            'true_label': row['True_Label'],
            'predicted_label': row['Predicted_Label'],
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
            # Generate CSV path
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            csv_path = f"csv_data/{video_name}_mediapipe_landmarks.csv"
            
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
        if not self.error_cases:
            return
        
        # Analyze error patterns
        error_labels = [case['true_label'] for case in self.error_cases]
        predicted_labels = [case['predicted_label'] for case in self.error_cases]
        
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
        if not self.error_cases:
            return
        
        # Extract confidence and motion data
        confidences = []
        motion_velocities = []
        
        for case in self.error_cases:
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
        
        error_videos_dir = f'{self.output_dir}/error_videos'
        os.makedirs(error_videos_dir, exist_ok=True)
        
        for i, case in enumerate(self.error_cases):
            try:
                video_name = case['video_name']
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
        summary_stats = {
            'total_cases': len(self.acoustic_results),
            'error_cases': len(self.error_cases),
            'success_cases': len(self.success_cases),
            'error_rate': len(self.error_cases) / len(self.acoustic_results) * 100,
            'success_rate': len(self.success_cases) / len(self.acoustic_results) * 100
        }
        
        with open(f'{self.output_dir}/summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # 2. Detailed error analysis
        error_details = []
        for case in self.error_cases:
            error_details.append({
                'video_name': case['video_name'],
                'true_label': case['true_label'],
                'predicted_label': case['predicted_label'],
                'confidence': case['confidence'],
                'motion_analysis': case.get('landmark_analysis', {}),
                'video_metadata': case.get('video_metadata', {})
            })
        
        error_df = pd.DataFrame(error_details)
        error_df.to_csv(f'{self.output_dir}/detailed_error_analysis.csv', index=False)
        
        # 3. Comparative analysis report
        comparison_report = {
            'motion_analysis': self.motion_analysis,
            'visibility_analysis': self.visibility_analysis,
            'temporal_analysis': self.temporal_analysis
        }
        
        with open(f'{self.output_dir}/comparative_analysis.json', 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        # 4. Key findings report
        self.generate_key_findings_report()
        
        logging.info("Analysis reports saved")
    
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
            f.write(f"- Total cases analyzed: {len(self.acoustic_results)}\n")
            f.write(f"- Error cases: {len(self.error_cases)} ({len(self.error_cases)/len(self.acoustic_results)*100:.1f}%)\n")
            f.write(f"- Success cases: {len(self.success_cases)} ({len(self.success_cases)/len(self.acoustic_results)*100:.1f}%)\n\n")
            
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
    logging.info("Starting Comprehensive Error Analysis")
    logging.info("="*50)
    
    # Initialize error analysis system
    analyzer = ErrorAnalysisSystem(acoustic_results_path, video_dataset_path, output_dir)
    
    # Run comprehensive analysis
    analyzer.analyze_error_patterns()
    
    logging.info("="*50)
    logging.info("Error Analysis Completed Successfully!")
    logging.info(f"Results saved in: {output_dir}")
    logging.info("="*50)

if __name__ == "__main__":
    main() 