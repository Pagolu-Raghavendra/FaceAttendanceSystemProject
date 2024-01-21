import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,
                              {
                                  'databaseURL':"Google Bucket URL"
                              }
                              )
ref=db.reference('Students')
data={
"22BQ5A0517":
        {
            "name":" Teja Munnangi",
            "class":"3-CSE-C",
            "total_attendance":8,
            "last_attendance_time":"2023-12-25 11:24:17"
        }}
for key,value in data.items():
    ref.child(key).set(value)
print("inserted")
