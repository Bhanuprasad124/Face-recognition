import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images('C:/Deeplearning/genenv/Data science/Deep Learning/project 1/images')

person_info={
    'ME':{"age": 30, "occupation": "Data Scientist"},
    'virat kohli': {"age": 30, "occupation": "Data Scientist"},
    'ms dhoni': {"age": 30, "occupation": "mann"}
    
}
# # # Load the CSV file containing person information
# person_info_df = pd.read_csv('person_info.csv')

# # Function to get person info from the CSV file
# def get_person_info(name):
#     # Search for the person by name in the CSV file
#     person = person_info_df[person_info_df['name'] == name]
    
#     # If person is found, return their details
#     if not person.empty:
#         return {"age": person['age'].values[0], "occupation": person['occupation'].values[0]}
#     else:
#         return {"age": "Unknown", "occupation": "Unknown"}

cap = cv2.VideoCapture(0)

while True:
    ret, frame=cap.read()

    face_locations,face_names=sfr.detect_known_faces(frame)
    
    for face_loc,name in zip(face_locations,face_names):
        top,right,bottom,left=face_loc[0],face_loc[1],face_loc[2],face_loc[3]
        
                # Get additional info about the person (if available)
        info = person_info.get(name, {})
        age = info.get("age", "Unknown")
        occupation = info.get("occupation", "Unknown")

        display_text = f"{name}, Age: {age}, {occupation}"

        cv2.putText(frame,display_text,(left,top-10),cv2.FONT_HERSHEY_DUPLEX,1,(0,0,200),2)
        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,200),2)

        
    cv2.imshow('Frame',frame)

    key=cv2.waitKey(1)
    if key==20:
        break
cap.release()
cap.destroyAllWindows()