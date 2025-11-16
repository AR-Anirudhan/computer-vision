import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from face_shape_detector import FaceShapeDetector
from recommendation_engine import GlassesRecommendationEngine
from utils import overlay_transparent

# ============================================================================
# KALMAN FILTER FOR SMOOTH TRACKING
# ============================================================================
class KalmanFilter:
    """Simple Kalman filter for 2D position smoothing"""
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                  [0, 1, 0, 1],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
    def update(self, measurement):
        """Update filter with new measurement"""
        self.kalman.correct(np.array([[np.float32(measurement[0])],
                                      [np.float32(measurement[1])]]))
        prediction = self.kalman.predict()
        return (int(prediction[0]), int(prediction[1]))


# ============================================================================
# FACE SHAPE DETECTOR
# ============================================================================
class FaceShapeDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh with 3D support
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.shape_classes = ['heart', 'oblong', 'oval', 'round', 'square']
        
        # Smoothing filters for stable tracking
        self.smoothing_window = 5
        self.position_history = deque(maxlen=self.smoothing_window)
        self.angle_history = deque(maxlen=self.smoothing_window)
        self.scale_history = deque(maxlen=self.smoothing_window)
        
        # Kalman filters for ultra-smooth tracking
        self.left_eye_filter = KalmanFilter()
        self.right_eye_filter = KalmanFilter()
    
    def detect_face(self, frame):
        """Detect face and return facial landmarks with 3D coordinates"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to pixel coordinates with 3D info
        h, w = frame.shape[:2]
        points_3d = np.array([[lm.x * w, lm.y * h, lm.z * w] 
                              for lm in face_landmarks.landmark], dtype=np.float32)
        
        return points_3d, results
    
    def classify_face_shape(self, landmarks):
        """Classify face shape using geometric measurements"""
        
        # Key MediaPipe landmark indices
        left_cheek = landmarks[234][:2]
        right_cheek = landmarks[454][:2]
        chin = landmarks[152][:2]
        forehead_top = landmarks[10][:2]
        
        left_jaw = landmarks[172][:2]
        right_jaw = landmarks[397][:2]
        
        left_temple = landmarks[21][:2]
        right_temple = landmarks[251][:2]
        
        left_cheekbone = landmarks[234][:2]
        right_cheekbone = landmarks[454][:2]
        
        # Calculate measurements
        face_width = np.linalg.norm(left_cheek - right_cheek)
        face_height = np.linalg.norm(chin - forehead_top)
        jaw_width = np.linalg.norm(left_jaw - right_jaw)
        forehead_width = np.linalg.norm(left_temple - right_temple)
        cheekbone_width = np.linalg.norm(left_cheekbone - right_cheekbone)
        
        # Calculate ratios
        aspect_ratio = face_height / face_width
        jaw_to_cheek = jaw_width / cheekbone_width
        jaw_to_forehead = jaw_width / forehead_width
        
        # Classification logic
        if aspect_ratio > 1.4:
            return 'oblong'
        elif aspect_ratio < 1.1:
            if jaw_to_forehead < 0.88:
                return 'heart'
            else:
                return 'round'
        else:
            if jaw_to_forehead < 0.82:
                return 'heart'
            elif jaw_to_forehead > 0.98:
                return 'square'
            else:
                return 'oval'
    
    def get_eye_landmarks_3d(self, landmarks):
        """Get eye positions with 3D coordinates and Kalman smoothing"""
        
        # MediaPipe eye indices
        left_eye_indices = [33, 133, 160, 158, 144, 153]
        right_eye_indices = [362, 263, 387, 385, 373, 380]
        
        # Calculate 3D eye centers
        left_eye_center = np.mean(landmarks[left_eye_indices], axis=0)
        right_eye_center = np.mean(landmarks[right_eye_indices], axis=0)
        
        # Apply Kalman filtering for smoothness
        left_eye_center[:2] = self.left_eye_filter.update(left_eye_center[:2])
        right_eye_center[:2] = self.right_eye_filter.update(right_eye_center[:2])
        
        # Get nose bridge for better alignment
        nose_bridge = landmarks[6]
        
        return left_eye_center, right_eye_center, nose_bridge
    
    def get_eye_landmarks(self, landmarks):
        """Get 2D eye positions for compatibility"""
        left_eye, right_eye, _ = self.get_eye_landmarks_3d(landmarks)
        return left_eye[:2].astype(int), right_eye[:2].astype(int)
    
    def get_smoothed_transform(self, left_eye, right_eye, scale):
        """Apply temporal smoothing to reduce jitter"""
        
        # Calculate center position
        center = ((left_eye[:2] + right_eye[:2]) / 2).astype(int)
        
        # Calculate angle
        angle = np.degrees(np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        ))
        
        # Add to history
        self.position_history.append(center)
        self.angle_history.append(angle)
        self.scale_history.append(scale)
        
        # Calculate smoothed values using moving average
        if len(self.position_history) > 0:
            smoothed_center = np.mean(self.position_history, axis=0).astype(int)
            smoothed_angle = np.mean(self.angle_history)
            smoothed_scale = np.mean(self.scale_history)
        else:
            smoothed_center = center
            smoothed_angle = angle
            smoothed_scale = scale
        
        return smoothed_center, smoothed_angle, smoothed_scale
    
    def draw_face_mesh(self, frame, results):
        """Draw the face mesh landmarks and connections on the frame."""
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
        return frame




# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================
class GlassesRecommendationEngine:
    """Provide glasses recommendations based on face shape"""
    
    def __init__(self):
        self.recommendations = {
            'heart': {
                'suitable': ['Round frames', 'Oval frames', 'Cat-eye', 'Rimless'],
                'avoid': ['Heavy top frames', 'Geometric shapes'],
                'tip': 'Look for frames wider at the bottom to balance your features. Light-colored or rimless styles, like aviators, can soften a wider forehead and draw attention downward.',
                'description': 'Heart-shaped faces have wider foreheads and narrow chins'
            },
            'oblong': {
                'suitable': ['Oversized frames', 'Square frames', 'Wide frames', 'Decorative temples'],
                'avoid': ['Small frames', 'Narrow styles'],
                'tip': 'Select frames with more depth than width to create an illusion of a shorter face. Decorative temples or bold, wide frames add width and balance your proportions.',
                'description': 'Oblong faces are longer than they are wide'
            },
            'oval': {
                'suitable': ['Most styles work!', 'Square', 'Rectangular', 'Geometric'],
                'avoid': ['Oversized frames that hide features'],
                'tip': 'Your balanced proportions work with most styles! Aim for frames as wide as the broadest part of your face. Avoid oversized frames that might overwhelm your natural symmetry.',
                'description': 'Oval faces have balanced proportions'
            },
            'round': {
                'suitable': ['Angular frames', 'Rectangular', 'Square', 'Cat-eye'],
                'avoid': ['Round frames', 'Small frames'],
                'tip': 'Create definition with strong, angular frames like rectangular or square shapes. A clear bridge can make your eyes appear wider, and geometric styles will sharpen your soft features.',
                'description': 'Round faces have full cheeks and soft curves'
            },
            'square': {
                'suitable': ['Round frames', 'Oval frames', 'Curved styles', 'Aviators'],
                'avoid': ['Angular frames', 'Boxy squares'],
                'tip': 'Soften your strong, angular features with round or oval frames. Thinner frames can also provide a nice contrast and prevent your look from becoming too heavy.',
                'description': 'Square faces have strong jawlines and broad foreheads'
            }
        }
    
    def get_recommendation(self, face_shape):
        """Get glasses recommendation for given face shape"""
        return self.recommendations.get(face_shape.lower(), {
            'suitable': ['Consult an optician'],
            'avoid': [],
            'tip': 'Face shape not recognized',
            'description': ''
        })


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def overlay_transparent(background, overlay, x, y):
    """Overlay transparent PNG on background image"""
    bg_h, bg_w = background.shape[:2]
    
    # Ensure overlay has alpha channel
    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    
    overlay_h, overlay_w = overlay.shape[:2]
    
    # Handle out of bounds
    if x >= bg_w or y >= bg_h or x + overlay_w <= 0 or y + overlay_h <= 0:
        return background
    
    # Calculate crop boundaries
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + overlay_w)
    y2 = min(bg_h, y + overlay_h)
    
    # Calculate overlay crop
    overlay_x1 = max(0, -x)
    overlay_y1 = max(0, -y)
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)
    
    # Extract regions
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    background_crop = background[y1:y2, x1:x2]
    
    # Check if regions are valid
    if overlay_crop.shape[0] == 0 or overlay_crop.shape[1] == 0:
        return background
    
    # Separate alpha channel and normalize
    overlay_rgb = overlay_crop[:, :, :3]
    alpha = overlay_crop[:, :, 3:] / 255.0
    
    # Blend images
    blended = (alpha * overlay_rgb + (1 - alpha) * background_crop).astype(np.uint8)
    
    # Copy back to background
    background[y1:y2, x1:x2] = blended
    
    return background


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class GlassesTryOnApp:
    def __init__(self):
        print("Initializing Glasses Try-On Application...")
        
        self.detector = FaceShapeDetector()
        self.recommender = GlassesRecommendationEngine()
        self.recommender = GlassesRecommendationEngine() # This class is now imported
        self.glasses_dict = self.load_glasses()
        self.current_shape = None
        
        # UI Settings
        self.sidebar_width = 350
        self.show_sidebar = True
        self.show_mesh = False
        
        # ADJUSTMENT PARAMETERS - Tune these for perfect fit
        self.glasses_scale = 2.1          # Width multiplier (2.0-2.5)
        self.vertical_offset = 0.08       # Up/down position (-0.2 to 0.2)
        self.horizontal_offset = 0        # Left/right position (-20 to 20 pixels)
        
    def load_glasses(self):
        """Load glasses images for each face shape"""
        glasses = {}
        shapes = ['heart', 'oblong', 'oval', 'round', 'square']
        
        for shape in shapes:
            shape_dir = f'assets/{shape}'
            if os.path.exists(shape_dir):
                png_files = [f for f in os.listdir(shape_dir) if f.endswith('.png')]
                if png_files:
                    img_path = os.path.join(shape_dir, png_files[0])
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        glasses[shape] = img
                        print(f"  ✓ {shape}: {png_files[0]}")
                
        return glasses
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Cannot access camera!")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("\n" + "="*60)
        print("GLASSES TRY-ON APPLICATION")
        print("="*60)
        print("\nControls:")
        print("  'q'     - Quit")
        print("  's'     - Re-detect face shape")
        print("  'h'     - Toggle sidebar")
        print("  'm'     - Toggle face mesh")
        print("  '1-5'   - Select shape")
        print("\n  Real-time Adjustments:")
        print("  'W/X'   - Move glasses Up/Down")
        print("  'A/D'   - Move glasses Left/Right")
        print("  'Z/C'   - Make glasses Smaller/Bigger")
        print("  'R'     - Reset adjustments")
        print("="*60 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect face and landmarks
            landmarks, results = self.detector.detect_face(frame)
            
            if landmarks is not None:
                # Draw face mesh if enabled
                # Draw face mesh if enabled (Note: FaceShapeDetector now handles drawing)
                if self.show_mesh:
                    frame = self.detector.draw_face_mesh(frame, results)

                # Detect face shape
                if self.current_shape is None:
                    self.current_shape = self.detector.classify_face_shape(landmarks)
                    print(f"✓ Detected: {self.current_shape.upper()}")
                
                # Overlay glasses if not showing mesh (to avoid clutter)
                # Overlay glasses
                if self.current_shape and self.current_shape in self.glasses_dict:
                    frame = self.overlay_glasses(frame, landmarks, 
                                                 self.glasses_dict[self.current_shape])
            
            # Add sidebar with info
            if self.show_sidebar:
                frame = self.draw_sidebar(frame, landmarks is not None)
            else:
                # Minimal overlay
                if self.current_shape:
                    cv2.putText(frame, f"{self.current_shape.upper()}", 
                               (20, 50), cv2.FONT_HERSHEY_DUPLEX, 
                               1.2, (0, 255, 0), 3)
            
            # Show frame
            cv2.imshow('Virtual Glasses Try-On', frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                self.current_shape = None
                print("Re-detecting face shape...")
            elif key == ord('h'):
                self.show_sidebar = not self.show_sidebar
            elif key == ord('m'):
                self.show_mesh = not self.show_mesh
            elif ord('1') <= key <= ord('5'):
                shapes = ['heart', 'oblong', 'oval', 'round', 'square']
                self.current_shape = shapes[key - ord('1')]
                print(f"✓ Set to: {self.current_shape.upper()}")
            
            # Real-time adjustment controls
            elif key == ord('w'):  # Move glasses UP
                self.vertical_offset -= 0.01
                print(f"Vertical offset: {self.vertical_offset:.2f}")
            elif key == ord('x'):  # Move glasses DOWN
                self.vertical_offset += 0.01
                print(f"Vertical offset: {self.vertical_offset:.2f}")
            elif key == ord('a'):  # Move glasses LEFT
                self.horizontal_offset -= 2
                print(f"Horizontal offset: {self.horizontal_offset}")
            elif key == ord('d'):  # Move glasses RIGHT
                self.horizontal_offset += 2
                print(f"Horizontal offset: {self.horizontal_offset}")
            elif key == ord('z'):  # Make glasses SMALLER
                self.glasses_scale -= 0.05
                print(f"Glasses scale: {self.glasses_scale:.2f}")
            elif key == ord('c'):  # Make glasses BIGGER
                self.glasses_scale += 0.05
                print(f"Glasses scale: {self.glasses_scale:.2f}")
            elif key == ord('r'):  # RESET to defaults
                self.glasses_scale = 2.1
                self.vertical_offset = 0.08
                self.horizontal_offset = 0
                print("✓ Reset to defaults")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def overlay_glasses(self, frame, landmarks, glasses_img):
        """Overlay glasses with adjustable parameters for perfect alignment"""
        if glasses_img is None:
            return frame
        
        try:
            # Get eye landmarks
            left_eye_indices = [33, 133, 160, 158, 144, 153]
            right_eye_indices = [362, 263, 387, 385, 373, 380]
            # Get smoothed 3D eye landmarks from the detector
            # The detector now handles Kalman filtering internally
            left_eye_3d, right_eye_3d, nose_bridge_3d = self.detector.get_eye_landmarks_3d(landmarks)
            left_eye = left_eye_3d[:2]
            right_eye = right_eye_3d[:2]
            
            left_eye_points = landmarks[left_eye_indices]
            right_eye_points = landmarks[right_eye_indices]
            
            # Calculate eye centers (2D)
            left_eye = np.mean(left_eye_points[:, :2], axis=0)
            right_eye = np.mean(right_eye_points[:, :2], axis=0)
            
            # Get nose bridge for centering
            nose_bridge = landmarks[6][:2]
            
            # Eye distance
            eye_distance = np.linalg.norm(right_eye - left_eye)
            
            # Calculate glasses size using adjustable scale
            glasses_width = int(eye_distance * self.glasses_scale)
            aspect_ratio = glasses_img.shape[0] / glasses_img.shape[1]
            glasses_height = int(glasses_width * aspect_ratio)
            
            # Resize
            glasses_resized = cv2.resize(glasses_img, (glasses_width, glasses_height))
            
            # Calculate center between eyes
            eye_center = ((left_eye + right_eye) / 2).astype(int)
            
            # Apply adjustments
            center_x = int(nose_bridge[0] + self.horizontal_offset)
            center_y = int(eye_center[1] + (glasses_height * self.vertical_offset))
            
            # Calculate rotation angle
            angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1],
                                         right_eye[0] - left_eye[0])) * -1
            
            # Rotate glasses
            glasses_rotated = self.rotate_image_with_alpha(glasses_resized, angle)
            
            # Final position
            x_offset = center_x - glasses_rotated.shape[1] // 2
            y_offset = center_y - glasses_rotated.shape[0] // 2
            
            # Overlay
            frame = overlay_transparent(frame, glasses_rotated, x_offset, y_offset)
            
        except Exception as e:
            pass
        
        return frame
    
    def rotate_image_with_alpha(self, image, angle):
        """Rotate image preserving transparency"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
        
        return rotated
    
    def draw_sidebar(self, frame, face_detected):
        """Draw information sidebar on the right side"""
        h, w = frame.shape[:2]
        
        # Create sidebar background with gradient
        sidebar = np.zeros((h, self.sidebar_width, 3), dtype=np.uint8)
        
        # Gradient effect
        for i in range(h):
            intensity = int(20 + (i / h) * 15)
            sidebar[i, :] = [intensity, intensity, intensity]
        
        # Add border
        cv2.line(sidebar, (0, 0), (0, h), (80, 80, 80), 2)
        
        y_offset = 30
        
        # Title
        cv2.putText(sidebar, "FACE ANALYSIS", 
                   (20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 
                   0.7, (255, 255, 255), 2)
        y_offset += 40
        
        # Separator line
        cv2.line(sidebar, (20, y_offset), (self.sidebar_width - 20, y_offset), 
                (100, 100, 100), 1)
        y_offset += 30
        
        # Face Shape Section
        cv2.putText(sidebar, "Face Shape:", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (200, 200, 200), 1)
        y_offset += 30
        
        if self.current_shape:
            shape_display = self.current_shape.upper()
            cv2.putText(sidebar, shape_display, 
                       (30, y_offset), cv2.FONT_HERSHEY_DUPLEX, 
                       1.0, (0, 255, 128), 2)
            
            icon = self.get_shape_icon(self.current_shape)
            cv2.putText(sidebar, icon, 
                       (self.sidebar_width - 60, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 128), 2)
        else:
            cv2.putText(sidebar, "Detecting...", 
                       (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX | cv2.FONT_ITALIC, 
                       0.7, (255, 255, 0), 1)
        
        y_offset += 50
        
        # Separator
        cv2.line(sidebar, (20, y_offset), (self.sidebar_width - 20, y_offset), 
                (100, 100, 100), 1)
        y_offset += 30
        
        # Recommendations Section
        if self.current_shape:
            rec = self.recommender.get_recommendation(self.current_shape)
            
            cv2.putText(sidebar, "RECOMMENDED STYLES", 
                       (20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 
                       0.6, (255, 255, 255), 2)
            y_offset += 35
            
            cv2.putText(sidebar, "Best Styles:", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (150, 255, 150), 1)
            y_offset += 25
            
            for style in rec['suitable'][:4]:
                cv2.circle(sidebar, (30, y_offset - 5), 3, (0, 255, 128), -1)
                
                if len(style) > 20:
                    style = style[:20] + "..."
                
                cv2.putText(sidebar, style, 
                           (45, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.45, (220, 220, 220), 1)
                y_offset += 25
            
            y_offset += 15
            
            cv2.putText(sidebar, "Avoid:", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (150, 150, 255), 1)
            y_offset += 25
            
            for style in rec['avoid'][:2]:
                cv2.putText(sidebar, "x", 
                           (27, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 100, 255), 2)
                
                if len(style) > 20:
                    style = style[:20] + "..."
                    
                cv2.putText(sidebar, style, 
                           (45, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.45, (220, 220, 220), 1)
                y_offset += 25
            
            y_offset += 20
            
            cv2.line(sidebar, (20, y_offset), (self.sidebar_width - 20, y_offset), 
                    (100, 100, 100), 1)
            y_offset += 25
            
            cv2.putText(sidebar, "PRO TIP", 
                       (20, y_offset), cv2.FONT_HERSHEY_DUPLEX, 
                       0.5, (100, 200, 255), 2)
            y_offset += 25
            
            tip_lines = self.wrap_text(rec['tip'], 30)
            for line in tip_lines:
                cv2.putText(sidebar, line, 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, (200, 200, 200), 1)
                y_offset += 20
        
        # Adjustment info
        y_bottom = h - 150
        cv2.line(sidebar, (20, y_bottom), (self.sidebar_width - 20, y_bottom), 
                (100, 100, 100), 1)
        y_bottom += 20
        
        cv2.putText(sidebar, "ADJUSTMENTS", 
                   (20, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        y_bottom += 20
        
        cv2.putText(sidebar, f"Scale: {self.glasses_scale:.2f}", 
                   (20, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (150, 150, 150), 1)
        y_bottom += 18
        
        cv2.putText(sidebar, f"Vertical: {self.vertical_offset:.2f}", 
                   (20, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (150, 150, 150), 1)
        y_bottom += 18
        
        cv2.putText(sidebar, f"Horizontal: {self.horizontal_offset}", 
                   (20, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, (150, 150, 150), 1)
        
        # Status indicator
        y_bottom = h - 55
        cv2.line(sidebar, (20, y_bottom), (self.sidebar_width - 20, y_bottom), 
                (100, 100, 100), 1)
        y_bottom += 20
        
        status_text = "Face Detected" if face_detected else "No Face"
        status_color = (0, 255, 0) if face_detected else (0, 0, 255)
        cv2.circle(sidebar, (30, y_bottom - 5), 6, status_color, -1)
        cv2.putText(sidebar, status_text, 
                   (45, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, status_color, 1)
        
        y_bottom += 20
        cv2.putText(sidebar, "W/X/A/D/Z/C to adjust", 
                   (20, y_bottom), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.35, (150, 150, 150), 1)
        
        # Combine frame and sidebar
        combined = np.hstack([frame, sidebar])
        
        return combined
    
    def get_shape_icon(self, shape):
        """Get icon for face shape"""
        icons = {
            'heart': '<3',
            'oval': 'O',
            'round': '@',
            'square': '[]',
            'oblong': '||'
        }
        return icons.get(shape, '?')
    
    def wrap_text(self, text, max_chars):
        """Wrap text into multiple lines"""
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        
        if current_line:
            lines.append(current_line.strip())
        
        return lines


# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    try:
        app = GlassesTryOnApp()
        app.run()
    except KeyboardInterrupt:
        print("\n✓ Application stopped")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
