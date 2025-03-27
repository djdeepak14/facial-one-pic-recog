import cv2
from simple_facerec import SimpleFaceRec

# Initialize the face recognizer
sfr = SimpleFaceRec()

# Load and encode a known face with correct path
sfr.encode_face('/Users/deepakkhanal/My_projects/messi.webp', "Messi")

# Open webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Detect and recognize faces
    face_locations, face_names, annotated_frame = sfr.detect_and_recognize(frame)
    
    # Print face locations (optional)
    for face_loc in face_locations:
        print(face_loc)
    
    # Show the annotated frame
    cv2.imshow("Frame", annotated_frame)
    
    # Exit on ESC key
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()