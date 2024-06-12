import os
import cv2
from flask import Flask, request, render_template, redirect, url_for
from datetime import date
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

MESSAGE = ""
# date2 = date.today().strftime("%d_%m_%y")
date2 = date.today().strftime("%d-%B-%Y")

detectingface = cv2.CascadeClassifier(
    'C:/Users/navin/OneDrive/Desktop/project/haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(0)
except:
    cap = cv2.VideoCapture(1)

if not os.path.isdir('Attendence'):
    os.makedirs('Attendence')
if not os.path.isdir('modelfile'):
    os.makedirs('modelfile')
if not os.path.isdir('modelfile/faces'):
    os.makedirs('modelfile/faces')
if f'Attendence-{date2}.csv' not in os.listdir('Attendence'):
    with open(f'Attendence/Attendence-{date2}.csv', 'w') as f:
        f.write('Name,ID,Time,Date')


def totalregistration():
    return len(os.listdir('modelfile/faces'))


def get_faces(image):
    if image != []:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_points = detectingface.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []


def identify_face(facearray):
    model = joblib.load('modelfile/model.pkl')
    return model.predict(facearray)


def model_taining():
    faces = []
    labels = []
    userlist = os.listdir('modelfile/faces')
    for user in userlist:
        for imgname in os.listdir(f'modelfile/faces/{user}'):
            img = cv2.imread(f'modelfile/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'modelfile/model.pkl')


def get_attendence():
    df = pd.read_csv(f'Attendence/Attendence-{date2}.csv')
    name = df['Name']
    id = df['ID']
    time = df['Time']
    datee = df['Date']
    l = len(df)
    return name, id, time, datee, l


def set_attendence(name):
    empname = name.split('_')[0]
    empid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    today_date = date.today().strftime("%d-%B-%y")

    df = pd.read_csv(f'Attendence/Attendence-{date2}.csv')
    if str(empid) not in list(df['ID']):
        with open(f'Attendence/Attendence-{date2}.csv', 'a') as f:
            f.write(f'\n{empname},{empid},{current_time},{today_date}')
    else:
        print("Exit ")
        


@app.route('/')
def homepage():
    name, id, time, datee, l = get_attendence()
    return render_template('home.html', name=name, id=id, time=time, datee=datee, l=l,
                           totalregistration=totalregistration(), date2=date2, mess=MESSAGE)


@app.route('/attendence', methods=['GET'])
def Attendence():
    ATTENDENCE = False
    if 'model.pkl' not in os.listdir('modelfile'):
        name, id, time, datee, l = get_attendence()
        MESSAGE = 'INVALID'
        print("INVALID")
        return render_template('home.html', name=name, id=id, time=time, datee=datee, l=l,
                               totalregistration=totalregistration, date2=date2, mess=MESSAGE)

    cap = cv2.VideoCapture(0)
    ret = True
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detectingface.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            personidentification = identify_face(face.reshape(1, -1))[0]

            if cv2.waitKey(1) == ord('a'):
                set_attendence(personidentification)
                current_time_ = datetime.now().strftime("%H:%M:%S")
                ATTENDENCE = True
                break
        if ATTENDENCE:
            print("attendence marked")
            break

        cv2.imshow('Mark attendence', frame)
        if cv2.waitKey(1) == ord('c'):
            break

    cap.release()
    cv2.destroyAllWindows()
    name, id, time, datee, l = get_attendence()
    MESSAGE = 'attendence marked successfully'

    return render_template('home.html', name=name, id=id, time=time, datee=datee, l=l,
                           totalregistration=totalregistration(),
                           date2=date2, mess=MESSAGE)


@app.route('/addEmployee', methods=['GET', 'POST'])
def addEmployee():
    newempname = request.form['newempname']
    newempid = request.form['newempid']
    userimagefolder = 'modelfile/faces/' + newempname + '_' + str(newempid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i = 0
    j = 0
    while 1:
        _, frame = cap.read()
        faces = get_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)

            if j % 10 == 0:
                name = newempname + '_' + str(i) + '.jpg'
                cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
                i += 1
            j += 1
        if j == 500:
            break
        cv2.imshow('Adding new  Employee', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    model_taining()
    name, id, time, datee, l = get_attendence()
    if totalregistration() > 0:
        name, id, time, datee, l = get_attendence()
        MESSAGE = 'Employee added Successfully'
        return render_template('home.html', name=name, id=id, time=time, datee=datee, l=l,
                               totalregistration=totalregistration(), date2=date2, mess=MESSAGE)
    else:
        return redirect(
            url_for('home.html', name=name, id=id, time=time, datee=datee, l=l, totalregistration=totalregistration(),
                    date2=date2))


# @app.route('/admin', methods=['GET'])
# def admin():
#     name,id,time,datee,l = get_attendence()
#     return render_template('home.html',name=name,id=id,time=time,datee=datee,l=l,totalregistration=totalregistration(),date2=date2, mess = MESSAGE)

app.run(debug=True, port=1000)
if __name__ == '__main__':
    pass
