import os
import time
from flask_cors import CORS
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
# from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
from test2 import haversine_distance, get_current_location
from geopy.geocoders import Nominatim
import geocoder
import dlib
from match_image_with_db import *
import threading



app = Flask(__name__)
CORS(app)

def db_con():
    conn = mysql.connector.connect(
        host="162.241.120.118",
        user="cognisun_all",
        password="Cognisun@456",
        database="NetworkSSOUAT"
    )
    print("DataBase Connected")
    return conn


# Facial recognition model
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read("face_recognizer_model.xml")



def capture_photos(member_id):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return jsonify({"error": "Could not open webcam"}), 500

    detector = dlib.get_frontal_face_detector()
    photos = []
    photos_binary = []
    count = 0

    if not os.path.exists(f'photos/{member_id}'):
        os.makedirs(f'photos/{member_id}')

    instructions = [
        "Look straight",
        "Turn your head slightly to the left",
        "Turn your head slightly to the right",
        "Look up",
        "Look down",
        "Tilt your head slightly to the left",
        "Tilt your head slightly to the right",
        "Look over your left shoulder",
        "Look over your right shoulder",
        "Look straight and smile"
    ]

    for instruction in instructions:
        while True:
            ret, frame = cap.read()
            if not ret:
                return jsonify({"error": "Failed to capture image"}), 501

            frame_with_instruction = frame.copy()
            frame_with_instruction = cv2.putText(frame_with_instruction, instruction, (10, 30),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Capturing Photos', frame_with_instruction)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return jsonify({"error": "Capture cancelled"}), 500

            cv2.waitKey(3000)

            # Capture the frame after the delay
            ret, frame = cap.read()
            if not ret:
                return jsonify({"error": "Failed to capture image"}), 501

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if faces:
                for face in faces:
                    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                    face_img = frame[y:y + h, x:x + w]

                    #TODO In binary
                    photos_binary.append(cv2.imencode('.jpg', face_img)[1].tobytes())

                    photo_path = f'photos/{member_id}/{member_id}_photo_{count + 1}.jpg'
                    cv2.imwrite(photo_path, face_img)
                    photos.append(photo_path)
                    count += 1
                    break
                break  # Exit the while loop to move to the next instruction

    cap.release()
    cv2.destroyAllWindows()
    return photos_binary, photos



@app.route("/update_photo/<member_id>", methods=['POST'])
def update_photos(member_id):
    photos_binary, photos = capture_photos(member_id)
    if len(photos) < 10:
        return jsonify({"message": "Failed to capture all required photos"}), 503

    face_labels = [member_id] * len(photos)
    
    # Train the LBPH face recognizer model
    # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    # try:
    #     if len(photos_binary) > 0:
    #         gray_images = [cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_GRAYSCALE) for photo in
    #                        photos_binary]
    #         # gray_images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in photos]

    #         print(gray_images)
    #         face_recognizer.train(gray_images, np.array(face_labels))
    # except cv2.error as e:
    #     return jsonify({"message": f"Error training face recognizer model: {e}"}), 500
    # face_recognizer.save("face_recognizer_model.xml")


    # Update profile_pics table
    conn = db_con()
    cursor = conn.cursor()

    try:
        query = "DELETE FROM profile_pics WHERE user_id = %s"
        cursor.execute(query, (member_id,))
        conn.commit()
    except:
        return jsonify({"message": "Error deleting profile pictures"}), 500


    # TODO Insert profile pictures
    query = "INSERT INTO profile_pics (user_id, photo1, photo2, photo3, photo4, photo5, photo6, photo7, photo8, photo9, photo10) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    params = [member_id, *photos_binary]
    cursor.execute(query, params)
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"id": member_id, "message": "Photos Updated successfully"}), 201



@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    try:name = data["name"]
    except: name = request.args.get("name")

    try:email = data["email"]
    except: email = request.args.get("email")

    try:mobile = data["mobile"]
    except: mobile = request.args.get("mobile")

    if name is None or email is None or mobile is None:
        return jsonify({"message": "Missing required parameters"}), 400

    print(name, email, mobile)

    conn = db_con()
    cursor = conn.cursor()
    query = f"INSERT INTO Members (`Name`, `Email`, `Mobile`) VALUES ('{name}', '{email}', '{mobile}')"
    print(query)
    cursor.execute(query)

    member_id = cursor.lastrowid
    print(member_id)

    photos_binary, photos = capture_photos(member_id)
    if len(photos) < 10:
        return jsonify({"message": "Failed to capture all required photos"}), 500

    face_labels = [member_id] * len(photos)

    # print("Photos : ", photos_binary)
    # print("Face Labels : ", face_labels)


    # Train the LBPH face recognizer model
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        if len(photos_binary) > 0:
            gray_images = [cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_GRAYSCALE) for photo in
                           photos_binary]
            face_recognizer.train(gray_images, np.array(face_labels))
    except cv2.error as e:
        return jsonify({"message": f"Error training face recognizer model: {e}"}), 500
    face_recognizer.save("face_recognizer_model.xml")


    # TODO Insert profile pictures
    query = "INSERT INTO profile_pics (user_id, photo1, photo2, photo3, photo4, photo5, photo6, photo7, photo8, photo9, photo10) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    params = [member_id, *photos_binary]
    cursor.execute(query, params)
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"id": member_id, "message": "User created successfully"}), 201


# @app.route("/create_venue", methods=["POST"])
# def create_venue():
#     data = request.get_json()
#     ven_name = data["ven_name"]
#     address = data["address"]
#     ven_lat = data["ven_lat"]
#     ven_long = data["ven_long"]

#     conn = db_con()
#     cursor = conn.cursor()

#     query = "INSERT INTO Venues (ven_name, address, ven_lat, ven_long) VALUES (%s, %s, %s, %s)"
#     cursor.execute(query, (ven_name, address, ven_lat, ven_long))
#     conn.commit()

#     return jsonify({"message": "Venue created successfully"}), 201


# @app.route("/create_meeting", methods=["POST"])
# def create_meeting():
#     data = request.get_json()
#     venue_id = data["venue_id"]
#     date = data["date"]
#     start_time = data["start_time"]
#     end_time = data["end_time"]

#     conn = db_con()
#     cursor = conn.cursor()

#     #TODO Check Venue_ID
#     ven_check_query = f"SELECT * FROM Venues WHERE venue_id = {venue_id}"
#     cursor.execute(ven_check_query)
#     venue = cursor.fetchone()
#     if not venue:
#         return jsonify({"message": "Venue not found"}), 404

#     query = "INSERT INTO Meetings (venue_id, date, start_time, end_time) VALUES (%s, %s, %s, %s)"
#     cursor.execute(query, (venue_id, date, start_time, end_time))
#     conn.commit()

#     return jsonify({"message": "Meeting created successfully"}), 201


# def fetch_meetings():
#     conn = db_con()
#     cursor = conn.cursor()
#     query = "SELECT * FROM Meetings"
#     cursor.execute(query)
#     meetings = cursor.fetchall()
#     meeting_list = []
#     for meeting in meetings:
#         meeting_list.append({
#             "id": meeting[0],
#             "title": meeting[1],
#             "date": meeting[2],
#             "start_time": meeting[3],
#             "end_time": meeting[4]
#         })
#     return meeting_list


# @app.route("/api/meetings", methods=["GET"])
# def get_meetings():
#     meeting_list = fetch_meetings()
#     return jsonify(meeting_list)


# @app.route("/", methods=["GET", "POST"])
# def dashboard():
    meetings = fetch_meetings()
    for meeting in meetings:
        if isinstance(meeting['end_time'], timedelta):
            # Assuming end_time is the time of the day, extract hours and minutes
            hours, remainder = divmod(meeting['end_time'].seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            meeting['end_time'] = datetime.strptime(f"{hours}:{minutes}", '%H:%M').time()

    current_time = datetime.now().time()
    today = datetime.now().date()
    return render_template("dashboard.html", meetings=meetings, current_time=current_time, today=today)


@app.route("/take_attendance/<venue_id>/<user_id>/<lat>/<long>", methods=["POST"])
def take_attendance(venue_id, user_id, lat, long):
    # user_id = input("Enter user ID: ")
    conn = db_con()
    cursor = conn.cursor()
    query = "SELECT meetingid,venueid,meetingdate,meetingstarttime,meetingendtime  FROM meeting WHERE meetingid = %s"
    
    cursor.execute(query, (venue_id,))
    meeting = cursor.fetchone()
    if meeting:
        timestamp = datetime.now()
        meeting_id = meeting[0]
        venue_id = meeting[1]
        meeting_date = meeting[2]
        start_time = meeting[3]
        end_time = meeting[4]
      
        # Ensure start_time is a datetime.time object
        if isinstance(end_time, timedelta):
            end_time = (datetime.min + end_time).time()

       
        # Ensure start_time is a datetime.time object
        if isinstance(start_time, timedelta):
            start_time = (datetime.min + start_time).time()

        # Combine the date and start time into a single datetime object
        start_datetime = datetime.combine(meeting_date, start_time)
        end_datetime = datetime.combine(meeting_date, end_time)

        if end_datetime > timestamp >= start_datetime - timedelta(minutes=30):

            #TODO Check Lat Long is within limit
            # latitude, longitude = get_current_location()
            # g = geocoder.ip('me')
            # latitude, longitude = g.latlng
            # latitude = request.form['lat']
            # longitude = request.form['long']
            latitude = float(lat)
            longitude = float(long)
            print("Your Lat-long",latitude, longitude)
            print("timestamp",timestamp)
            # if latitude is None or longitude is None:
            #     return jsonify({"message": "Error fetching location"}), 500
            # elif latitude == 429:
            #     return jsonify({"status_code":429, "message": "Error fetching location. Too Many Requests"}), 500

            venue_q = f"SELECT latitude, longitude FROM address WHERE addressid = '{venue_id}'"
            print(venue_q)
            cursor.execute(venue_q)
            venue = cursor.fetchone()
            if not venue:
                return jsonify({"message": "Venue not found"}), 404
            else:
                venue_lat = venue[0]
                venue_long = venue[1]
                print("Venue Lat-Long : ", venue_lat, venue_long)

            distance = haversine_distance(latitude, longitude, venue_lat, venue_long)
            print("Distance from venue:", distance)
            if distance > 50:
                return jsonify({"message": "Geolocation is not within the configured distance", "Distance in Meter": distance, "Note": "The distance limit is 50 meter"}), 401


            qu = f"SELECT MemberID, MeetingID, attendancecode from attendance where MemberID = '{user_id}' and MeetingID = '{meeting_id}'"
            cursor.execute(qu)
            result = cursor.fetchone()
            if result:
                if result[2] == "A":
                # return jsonify({"message": "Attendance already taken"}), 403
                    print("Taking attendance...")

                    reset_known_faces()
                    update_known_faces(user_id)
                    result = process_frame()
                    print("Result : ", result)
                    if str(user_id) not in result or "LIVE" not in result:
                        return jsonify({"message": "Face Not recognized or Member Not Found"})
                    else:
                        if result:
                            if timestamp > start_datetime + timedelta(minutes=30):
                                status = "L"
                            else:
                                status = "P"



                            query = f"UPDATE attendance set attendancecode='{status}' where MemberID = '{user_id}' and MeetingID = '{meeting_id}'"

                            # query = "INSERT INTO attendance (user_id, meeting_id, timestamp, latitude, longitude, status) VALUES (%s, %s, %s, %s, %s, %s)"
                            # cursor.execute(query, (user_id, meeting_id, datetime.now(),latitude, longitude, status))  # Present
                            cursor.execute(query)
                            conn.commit()
                            return jsonify({"message": "Attendance taken successfully"}), 200

                        else:
                            return jsonify({"message": "Face Not recognized or Member Not Found"})
                else:
                    return jsonify({"message": "Attendance already taken"}), 403
            # else:
            #     return jsonify({"message": "Attendance already taken"}), 403


            

        elif timestamp > end_datetime:
            return jsonify({"message": "Meeting has ended"}), 403
        else:
            return jsonify({"message": "Meeting has not started yet"}), 402
    else:
        return jsonify({"message": "Meeting not found"}), 404
    
@app.route("/check_photos/<user_id>", methods=['GET','POST'])
def check_photos(user_id):
    conn = db_con()
    cursor = conn.cursor()
    
    query = "SELECT * FROM profile_pics WHERE user_id = %s"
    cursor.execute(query, (user_id,))
    photos = cursor.fetchall()
    if photos:
        return jsonify({"ID":f"{user_id}","message": "Photos found"}), 200
    else:
        return jsonify({"message": "No photos found"}), 404


@app.route('/')
def start_api():
    return "API Calling"

@app.route('/hello')
def home():
    return jsonify(message="Hello, world!")

if __name__ == "__main__":
    #   app.run(debug=False)        #for debub
     app.run(debug=True)        # for Run
