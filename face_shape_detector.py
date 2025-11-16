import cv2
import numpy as np
import mediapipe as mp

class FaceShapeDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.shape_classes = ['heart', 'oblong', 'oval', 'round', 'square']
    
    def detect_face(self, frame):
        """Detect face and return facial landmarks (468 points)"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to pixel coordinates
        h, w = frame.shape[:2]
        points = np.array([[int(lm.x * w), int(lm.y * h)] 
                          for lm in face_landmarks.landmark])
        
        return points
    
    def classify_face_shape(self, landmarks):
        """Classify face shape using geometric measurements"""
        
        # Key MediaPipe landmark indices (468-point model)
        # Face contour
        left_cheek = landmarks[234]      # Left face boundary
        right_cheek = landmarks[454]     # Right face boundary
        chin = landmarks[152]            # Bottom of chin
        forehead_top = landmarks[10]     # Top of face
        
        # Jawline
        left_jaw = landmarks[172]        # Left jaw
        right_jaw = landmarks[397]       # Right jaw
        
        # Forehead/temple
        left_temple = landmarks[21]      # Left temple
        right_temple = landmarks[251]    # Right temple
        
        # Cheekbones
        left_cheekbone = landmarks[234]
        right_cheekbone = landmarks[454]
        
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
        
        # Classification logic based on facial proportions
        if aspect_ratio > 1.4:
            # Long face
            return 'oblong'
        elif aspect_ratio < 1.1:
            # Wide face
            if jaw_to_forehead < 0.88:
                return 'heart'
            else:
                return 'round'
        else:
            # Balanced proportions
            if jaw_to_forehead < 0.82:
                # Narrow jaw
                return 'heart'
            elif jaw_to_forehead > 0.98:
                # Square jaw
                return 'square'
            else:
                # Balanced
                return 'oval'
    
    def get_eye_landmarks(self, landmarks):
        """Get eye center positions for glasses overlay"""
        
        # MediaPipe eye landmark indices
        # Left eye (from viewer's perspective - person's right eye)
        left_eye_indices = [33, 133, 160, 158, 144, 153]
        # Right eye (from viewer's perspective - person's left eye)
        right_eye_indices = [362, 263, 387, 385, 373, 380]
        
        # Calculate eye centers
        left_eye_center = np.mean(landmarks[left_eye_indices], axis=0).astype(int)
        right_eye_center = np.mean(landmarks[right_eye_indices], axis=0).astype(int)
        
        return left_eye_center, right_eye_center
    
    def draw_landmarks(self, frame, landmarks):
        """Draw facial landmarks for debugging (optional)"""
        for point in landmarks:
            cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)
        return frame
