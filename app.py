import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import re
import os
import time
import dlib
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import tkinter as tk
from tkinter import ttk

# Function to calculate Euclidean distance
def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)


# Load the pre-trained FaceNet model
model_path = 'facenet_model.pb'
with tf.io.gfile.GFile(model_path, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Load the model into a TensorFlow graph
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")

# Create a session to run the graph
with tf.compat.v1.Session(graph=graph) as sess:
    # Get the input and output tensors
    input_tensor = graph.get_tensor_by_name("input:0")
    output_tensor = graph.get_tensor_by_name("embeddings:0")
    phase_train_tensor = graph.get_tensor_by_name("phase_train:0")

    # Initialize dlib's face detector and facial landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    face_predictor = dlib.shape_predictor(predictor_path)

    # Load Haar Cascade for quick face detection
    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)

    # Load the saved embeddings and their labels
    embeddings_dict = {}
    student_folder = "students01"
    for student_name in os.listdir(student_folder):
        student_path = os.path.join(student_folder, student_name)
        if os.path.isdir(student_path):
            student_embeddings = []
            for file_name in os.listdir(student_path):
                if file_name.endswith("_embeddings.npy"):
                    embedding = np.load(os.path.join(student_path, file_name))
                    student_embeddings.append(embedding)
            embeddings_dict[student_name] = student_embeddings

    # Home page where users can choose to log in as Teacher or Admin
    def home_page():
        st.title("Attendance System")
        st.subheader("Welcome to the Attendance System")
        
        login_type = st.radio("Select your role", ["Teacher", "Admin"])
        
        if login_type == "Teacher":
            if st.button("Login as Teacher"):
                st.session_state.page = "teacher_login"
                st.experimental_rerun()  # Rerun to navigate to the teacher login page

        elif login_type == "Admin":
            if st.button("Login as Admin"):
                st.session_state.page = "admin_login"
                st.experimental_rerun()  # Rerun to navigate to the admin login page

    # Teacher Login page
    def teacher_login():
        st.title("Teacher Login")
        username = st.text_input("Enter your username")
        password = st.text_input("Enter your password", type='password')

        if st.button("Login"):
            if username == "teacher" and password == "password":  # Replace with real authentication logic
                st.session_state.logged_in_as = "teacher"
                st.session_state.page = "teacher_menu"
                st.experimental_rerun()  # Rerun to navigate to the teacher menu
            else:
                st.error("Invalid credentials")

    def admin_login():
        st.title("Admin Login")
        username = st.text_input("Admin Username")
        password = st.text_input("Admin Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "admin123":  # Simple hardcoded credentials for now
                st.session_state.logged_in_as = "admin"
                st.session_state.page = "admin_menu"
                st.experimental_rerun()  # Rerun to navigate to the teacher menu
            else:
                st.error("Invalid Admin credentials")

    # Teacher Menu page
    def teacher_menu():
        st.title("Teacher Menu")

        option = st.sidebar.selectbox("Select an Option", ["Take Attendance", "Update Attendance","Upload Image/Video"])

        if option == "Take Attendance":
            take_attendance()
        elif option == "Update Attendance":
            update_attendance()
        elif option == "Upload Image/Video":
            upload_media()

        if st.button("Logout"):
            st.session_state.logged_in_as = None
            st.session_state.page = "home"
            st.experimental_rerun()

    def admin_menu():
        st.title("Admin Menu")

        option = st.sidebar.selectbox("Select an Option", ["Add Student"])

        if option == "Add Student":
            add_student()

        if st.button("Logout"):
            st.session_state.logged_in_as = None
            st.session_state.page = "home"
            st.experimental_rerun()

    def get_existing_subjects():
        subject_files = [f for f in os.listdir() if f.endswith('_attendance.csv')]
        subjects = [f.split('_')[0] for f in subject_files]  # Extract subject name from file name
        return subjects

    def is_valid_email(email):
        email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        return re.match(email_regex, email) is not None

    def send_email(to_address, subject, message_body):
    # Email account details
        from_address = "khare.ritik01@gmail.com"  # Replace with your email
        password = "bpqz vqhy xydm utgb"  # Replace with your email password

        # Set up the MIME message
        msg = MIMEMultipart()
        msg['From'] = from_address
        msg['To'] = to_address
        msg['Subject'] = subject
        msg.attach(MIMEText(message_body, 'plain'))

        # Sending the email
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(from_address, password)
            server.send_message(msg)
            server.quit()
            print(f"Email sent to {to_address}")
        except Exception as e:
            print(f"Failed to send email to {to_address}. Error: {e}")


    def upload_media():
        st.title("Upload Media for Attendance")
        
        # Get existing subjects
        existing_subjects = get_existing_subjects()

        # Create a combined list of existing subjects and an option to input a new subject
        subject_options = existing_subjects + ["Enter a new subject"]

        # Dropdown to select or enter a new subject
        subject_choice = st.selectbox("Select or Enter Subject", subject_options)

        # If "Enter a new subject" is selected, display an input box to allow the user to type it
        if subject_choice == "Enter a new subject":
            subject = st.text_input("Enter the subject name (e.g., DSA, Python): ").strip()
        else:
            subject = subject_choice

        if subject:  # Proceed only if a subject is specified
            # Define the attendance file name based on the subject
            attendance_date = datetime.today().date()
            attendance_file = f"{subject}_attendance.csv"

            # Check if attendance has already been taken for this subject today
            attendance_taken = False
            if os.path.exists(attendance_file):
                with open(attendance_file, "r") as f:
                    lines = f.readlines()
                    for line in lines[1:]:  # Skip header
                        date, subject_in_record, *_ = line.strip().split(',')
                        if date == str(attendance_date) and subject_in_record == subject:
                            attendance_taken = True
                            break

            if attendance_taken:
                st.warning(f"Attendance for {subject} on {attendance_date} has already been taken.")
            else:
                # Prompt for media upload
                uploaded_file = st.file_uploader("Upload an image or video for attendance", type=["jpg", "png", "mp4", "avi"])
                if uploaded_file:
                    temp_file_path = f"temp_{uploaded_file.name}"
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.read())

                    # Create the attendance file with headers if it doesn't exist
                    if not os.path.exists(attendance_file):
                        with open(attendance_file, "w") as f:
                            f.write("date,subject,student_name,registration_number,email,attendance_status\n")

                    # Process the uploaded media
                    if st.button("Take Attendance"):
                        if uploaded_file.type.startswith("image"):
                            process_image(temp_file_path, attendance_file, subject, attendance_date)
                        elif uploaded_file.type.startswith("video"):
                            process_video(temp_file_path, attendance_file, subject, attendance_date)
                        else:
                            st.error("Unsupported file type. Please upload an image or video.")
                        
                        # Clean up temporary file
                        os.remove(temp_file_path)
    
    def recognize_face(face_img):
        face_img = cv2.resize(face_img, (160, 160))
        face_img = (face_img - 127.5) / 128.0
        return sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(face_img, axis=0), phase_train_tensor: False})

# Function to process uploaded image
    def process_image(image_path, attendance_file, subject, attendance_date):
        st.write("Processing uploaded image...")
        frame = cv2.imread(image_path)

        # Initialize attendance tracking variables
        recognized_students = set()
        total_recognized = 0
        total_unrecognized = 0

        # Process the image frame
        # process_frame(frame, recognized_students, total_recognized, total_unrecognized, attendance_file, subject, attendance_date)
        process_frame(frame, recognized_students, attendance_file, subject, attendance_date)

        # Display the processed image
        st.image(frame, channels="BGR")

    def process_video(video_path, attendance_file, subject, attendance_date):
    
    # Initialize OpenCV VideoCapture
        video_capture = cv2.VideoCapture(video_path)
        recognized_students = set()  # Track recognized students

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break  # End of video

            # Process the frame for face recognition and attendance
            frame = process_frame(frame, recognized_students, attendance_file, subject, attendance_date)

            # Display the frame in an OpenCV window
            cv2.imshow("Face Recognition Processing", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                st.success("Video processing stopped by the user. Attendance has been updated.")
                break

        video_capture.release()
        cv2.destroyAllWindows()  # Close OpenCV window
        print(f"Video processing completed. Attendance saved to {attendance_file}.")


    def process_frame(frame, recognized_students, attendance_file, subject, attendance_date):
    
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame, 1)

        for face in faces:
            # Align and preprocess face
            landmarks = face_predictor(gray_frame, face)
            aligned_face = dlib.get_face_chip(frame, landmarks, size=160)
            face_embedding = recognize_face(aligned_face)

            # Recognize the face
            min_distance = float("inf")
            recognized_name = "Unknown"

            for student_name, student_embedding in embeddings_dict.items():
                distance = euclidean_distance(face_embedding, student_embedding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_name = student_name

            # Apply recognition threshold
            threshold = 0.80
            if min_distance < threshold and recognized_name != "Unknown":
                # Extract student details
                student_data = recognized_name.split('_')
                student_name, reg_number, email = student_data[0], student_data[1], student_data[2]

                # Update attendance if not already recognized
                if recognized_name not in recognized_students:
                    recognized_students.add(recognized_name)
                    with open(attendance_file, "a") as f:
                        f.write(f"{attendance_date},{subject},{student_name},{reg_number},{email},Present\n")
                label = reg_number
            else:
                label = "Unknown"

            # Draw bounding box and label on the frame
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame


            
    # Take Attendance function with "Take Attendance" button
    def take_attendance():
        st.title("Take Attendance")
        true_labels = []
        predicted_labels = []

        # Get existing subjects from the database
        existing_subjects = get_existing_subjects()

        # Create a combined list of existing subjects and an option to input a new subject
        subject_options = existing_subjects + ["Enter a new subject"]

        # Dropdown to select or enter a new subject
        subject_choice = st.selectbox("Select or Enter Subject", subject_options)
        
        # If "Enter a new subject" is selected, display an input box to allow the user to type it
        if subject_choice == "Enter a new subject":
            subject = st.text_input("Enter the subject name (e.g., DSA, Python): ").strip()
        else:
            subject = subject_choice

        if subject:  # Proceed only if a subject is specified
            # Load student details from students_details.csv
            students_df = pd.read_csv('students_details.csv')
            
            # Attendance record CSV file for the subject
            attendance_date = datetime.today().date()
            attendance_file = f"{subject}_attendance.csv"
            
            # Check if attendance has already been taken for the subject today
            attendance_taken = False
            if os.path.exists(attendance_file):
                with open(attendance_file, "r") as f:
                    lines = f.readlines()
                    for line in lines[1:]:  # Skip header
                        date, subject_in_record, *_ = line.strip().split(',')
                        if date == str(attendance_date) and subject_in_record == subject:
                            attendance_taken = True
                            break

            # Display a button to start attendance taking
            if st.button("Take Attendance"):
                if attendance_taken:
                    st.warning(f"Attendance for {subject} on {attendance_date} has already been taken.")
                else:
                    # Create the CSV file with headers if it doesn't exist
                    if not os.path.exists(attendance_file):
                        with open(attendance_file, "w") as f:
                            f.write("date,subject,student_name,registration_number,email,attendance_status\n")
                    
                    st.success(f"Attendance for {subject} on {attendance_date} is being taken now.")
                    
                    # Initialize OpenCV VideoCapture
                    cap = cv2.VideoCapture(0)
                    recognized_students = set()  # Track recognized students by registration number

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to grab frame")
                            break

                        # Convert frame to grayscale for Haar Cascade detection
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        haar_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

                        # Convert frame to RGB for dlib
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        for (x, y, w, h) in haar_faces:
                            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                            landmarks = face_predictor(rgb_frame, dlib_rect)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            face_img = frame[y:y + h, x:x + w]
                            if face_img.size > 0:
                                face_img = cv2.resize(face_img, (160, 160))
                                face_img = (face_img - 127.5) / 128.0  # Normalize image

                                face_embedding = sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(face_img, axis=0), phase_train_tensor: False})

                                min_distance = float("inf")
                                recognized_name = "Unknown"
                                recognized_reg_number = None
                                for student_name, saved_embeddings in embeddings_dict.items():
                                    for saved_embedding in saved_embeddings:
                                        distance = euclidean_distance(face_embedding, saved_embedding)
                                        if distance < min_distance:
                                            min_distance = distance
                                            recognized_name = student_name

                                threshold = 0.599999  # Adjust this value based on your needs
                                if min_distance < threshold and recognized_name != "Unknown":
                                    label = recognized_name
                                    student_data = recognized_name.split('_')
                                    student_name, reg_number, email = student_data[0], student_data[1], student_data[2]

                                    if reg_number not in recognized_students:
                                        recognized_students.add(reg_number)

                                        # Record present status for the recognized student
                                        with open(attendance_file, "a") as f:
                                            f.write(f"{attendance_date},{subject},{student_name},{reg_number},{email},Present\n")
                                        
                                        true_labels.append(1)  # True label: Present
                                        predicted_labels.append(1)  # Predicted label: Recognized


                                        
                                        email_sent = False
                                        # print("its executed and working")
                                        
                                        if not email_sent:  # Check if email has already been sent
                                            email_subject = f"Attendance Confirmation for {subject}"
                                            email_body = f"Hello {student_name},\n\nYour attendance has been successfully recorded for {subject} on {attendance_date}.\n\nBest regards,\nAttendance System"
                                            if send_email(email, email_subject, email_body):
                                                print(f"Confirmation email sent to {email}.")
                                                email_sent = True  # Mark as email sent
                                            else:
                                                print(f"Failed to send email to {email}.")
                                else:
                                    # label = "Unknown"
                                    label = "Unknown"
                                    reg_number = "N/A"  # Set a default value for unrecognized faces
                                    true_labels.append(1)  # True label: Present
                                    predicted_labels.append(0)

                            # Display registration number above the face
                            cv2.putText(frame, reg_number, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                        # Display the frame in a window
                        cv2.imshow("Real-Time Face Recognition", frame)

                        # Press 'q' to quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    cap.release()
                    cv2.destroyAllWindows()

                    # Mark absent for all students who weren't recognized
                    unrecognized_students = students_df[~students_df['Registration Number'].isin(recognized_students)]
                    with open(attendance_file, "a") as f:
                        for _, row in unrecognized_students.iterrows():
                            f.write(f"{attendance_date},{subject},{row['Name']},{row['Registration Number']},{row['Student Email']},Absent\n")

                    st.success("Attendance recorded successfully.")
        
        # recognized_students = set()

        # # [Code for taking attendance and recognizing students...]

        # for recognized_name in recognized_students:
        #     # Extract student details
        #     student_data = recognized_name.split('_')
        #     student_name, reg_number, email = student_data[0], student_data[1], student_data[2]

        #     # Mark attendance in CSV
        #     with open(attendance_file, "a") as f:
        #         f.write(f"{attendance_date},{subject},{student_name},{reg_number},{email},Present\n")

        #     # Send confirmation email
        #     email_subject = f"Attendance Confirmation for {subject}"
        #     email_body = f"Hello {student_name},\n\nYour attendance has been successfully recorded for {subject} on {attendance_date}.\n\nBest regards,\nAttendance System"
        #     send_email(email, email_subject, email_body)

    def add_student():
        # Input fields for student details
        student_name = st.text_input("Enter the student's name:")
        registration_number = st.text_input("Enter the student's registration number:")
        student_email = st.text_input("Enter the student's email:")
        parent_email = st.text_input("Enter the parent's email:")

        if st.button("Start Face Registration"):
            # Validate email format
            if not is_valid_email(student_email):
                st.error("Invalid email format. Please re-enter the email.")
                return

            # Create a unique folder for the student
            student_folder_name = f"{student_name}_{registration_number}_{student_email}"
            student_folder = os.path.join("students", student_folder_name)
            if not os.path.exists(student_folder):
                os.makedirs(student_folder)

            # Save student information
            student_info_path = os.path.join(student_folder, "info.txt")
            with open(student_info_path, 'w') as info_file:
                info_file.write(f"Name: {student_name}\n")
                info_file.write(f"Registration Number: {registration_number}\n")
                info_file.write(f"Student Email: {student_email}\n")
                info_file.write(f"Parent Email: {parent_email}\n")

            # Set up video capture and face registration
            cap = cv2.VideoCapture(0)
            max_images = 100
            count = 0
            st.info("Please follow the instructions on the camera screen.")

            # Instructions and parameters for capturing
            instructions = ["Look straight", "Look up", "Look down", "Turn left", "Turn right"]
            instruction_index = 0
            instruction_change_time = 5
            last_change_time = time.time()
            
            # Zoom parameters
            zoom_factor = 1.0
            zoom_step = 0.01
            zoom_direction = 1

            # Capture images
            while count < max_images:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video.")
                    break

                # Apply zoom
                height, width = frame.shape[:2]
                center_x, center_y = width // 2, height // 2
                new_width = int(width / zoom_factor)
                new_height = int(height / zoom_factor)
                start_x = center_x - new_width // 2
                start_y = center_y - new_height // 2
                zoomed_frame = frame[start_y:start_y + new_height, start_x:start_x + new_width]
                frame = cv2.resize(zoomed_frame, (width, height))

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                haar_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                current_time = time.time()
                if current_time - last_change_time >= 0.5:  # Adjust capture delay if needed
                    for (x, y, w, h) in haar_faces:
                        if w > 0 and h > 0:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                            landmarks = face_predictor(rgb_frame, dlib_rect)

                            face_img = frame[y:y + h, x:x + w]
                            if face_img.size > 0:
                                face_img = cv2.resize(face_img, (160, 160))
                                face_img = (face_img - 127.5) / 128.0

                                embeddings = sess.run(output_tensor, feed_dict={input_tensor: np.expand_dims(face_img, axis=0), phase_train_tensor: False})
                                np.save(os.path.join(student_folder, f"{student_name}_{registration_number}_{count}_embeddings.npy"), embeddings)
                                cv2.imwrite(os.path.join(student_folder, f"{student_name}_{registration_number}_{count}.jpg"), face_img * 128.0 + 127.5)

                                count += 1
                                last_change_time = current_time

                # Display instruction and zoom effect
                cv2.putText(frame, instructions[instruction_index], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if current_time - last_change_time > instruction_change_time:
                    instruction_index = (instruction_index + 1) % len(instructions)
                    last_change_time = time.time()

                zoom_factor += zoom_direction * zoom_step
                if zoom_factor >= 1.5 or zoom_factor <= 1.0:
                    zoom_direction *= -1

                cv2.imshow("Add Student - Face Registration", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

            st.success(f"Student '{student_name}' added successfully with {count} images and embeddings.")

            # Append student details to CSV
            student_data = {
                "Name": student_name,
                "Registration Number": registration_number,
                "Student Email": student_email,
                "Parent Email": parent_email
            }
            df = pd.DataFrame([student_data])
            df.to_csv('students_details.csv', mode='a', header=False, index=False)

    # Update Attendance function
    def update_attendance():
        st.title("Update Attendance")

        # Ask for the subject
        existing_subjects = get_existing_subjects()

        # Create a combined list of existing subjects and an option to input a new subject
        subject_options = existing_subjects + ["Enter a new subject"]

        # Dropdown to select or enter a new subject
        subject_choice = st.selectbox("Select or Enter Subject", subject_options)
        
        # If "Enter a new subject" is selected, display an input box to allow the user to type it
        if subject_choice == "Enter a new subject":
            subject = st.text_input("Enter the subject name (e.g., DSA, Python): ").strip()
        else:
            subject = subject_choice

        # Ask for the date to check or update attendance (Using date_input for calendar view)
        date_input = st.date_input("Select the date to update attendance", min_value=datetime(2000, 1, 1), max_value=datetime.today().date())
        
        # Convert the date input to string in "YYYY-MM-DD" format
        date_str = date_input.strftime("%Y-%m-%d")

        # Check if the attendance file exists
        attendance_file = f"{subject}_attendance.csv"
        if not os.path.exists(attendance_file):
            st.error(f"No attendance records found for the subject: {subject}")
            return
        
        # Read the attendance records from the CSV file
        attendance_df = pd.read_csv(attendance_file)

        # Filter records by the selected date
        filtered_df = attendance_df[attendance_df['date'] == date_str]

        if filtered_df.empty:
            st.warning(f"No attendance records found for {subject} on {date_str}.")
        else:
            # Show the filtered attendance records
            st.write(f"Attendance records for {subject} on {date_str}:")
            st.dataframe(filtered_df)

            # Option to update attendance
            update_option = st.radio("Do you want to update attendance?", ("No", "Yes"))
            
            if update_option == "Yes":
                # Select a student to update
                student_names = filtered_df['student_name'].unique()
                selected_student = st.selectbox("Select student to update attendance", student_names)

                # Get the row for the selected student
                student_row = filtered_df[filtered_df['student_name'] == selected_student]
                 
                current_status = student_row['attendance_status'].values[0]

            # Display the current attendance status
                st.write(f"Current status of {selected_student}: {current_status}")
                # Update the attendance status (Present/Absent)
                new_status = st.selectbox("Update status", ("Present", "Absent"))
                
                # Update the status
                if st.button("Update Attendance"):
                    # Modify the status in the DataFrame
                    attendance_df.loc[attendance_df['student_name'] == selected_student, 'attendance_status'] = new_status
                    
                    # Save the updated DataFrame back to the CSV
                    attendance_df.to_csv(attendance_file, index=False)

                    st.success(f"Attendance for {selected_student} updated to {new_status}.")


    # Display the appropriate page based on login state
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home_page()
    elif st.session_state.page == "teacher_login":
        teacher_login()
    elif st.session_state.page == "teacher_menu":
        teacher_menu()
    elif st.session_state.page == "admin_login":
        admin_login()
    elif st.session_state.page == "admin_menu":
        admin_menu()
