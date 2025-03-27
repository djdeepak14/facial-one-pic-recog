import cv2
import face_recognition

class SimpleFaceRec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_image(self, image_path):
        """Load an image from a file path."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image at {image_path}")
        return img

    def encode_face(self, image_path, name=None):
        """Encode a face from an image and optionally store it with a name."""
        img = self.load_image(image_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)
        if not encodings:
            raise ValueError(f"No faces detected in {image_path}")
        encoding = encodings[0]  # Take the first detected face
        if name:
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
        return encoding

    def detect_and_recognize(self, frame):
        """Detect and recognize faces in a frame, returning locations, names, and annotated frame."""
        # Convert frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            face_names.append(name)
        
        # Draw rectangles and names on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return face_locations, face_names, frame
     
        

if __name__ == "__main__":
    sfr = SimpleFaceRec()
    sfr.encode_face('/Users/deepakkhanal/My_projects/messi.webp', "Messi")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_locations, face_names, annotated_frame = sfr.detect_and_recognize(frame)
        cv2.imshow("Frame", annotated_frame)
        if cv2.waitKey(1) == 27:  # ESC
            break
    cap.release()
    cv2.destroyAllWindows()