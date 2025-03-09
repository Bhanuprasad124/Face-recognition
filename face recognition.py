import cv2
import face_recognition

# Load the reference image (Virat Kohli) and encode it
img = cv2.imread('C:/Users/BHANU PRASAD/dataset_frames')

# Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
bgr_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get the face encoding for the reference image (Virat Kohli)
img_encode = face_recognition.face_encodings(bgr_img)[0]

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture the current frame from the webcam
    ret, frame = cap.read()

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]  # Convert from BGR to RGB

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Compare the detected faces with the reference image (Virat Kohli)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([img_encode], face_encoding)
        
        # Default to "Unknown"
        name = "Unknown"

        # If there's a match, assign the person's name (e.g., Virat Kohli)
        if matches[0]:
            name = "Virat Kohli"

        # Scale back up the face locations to match the original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Label the face
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
