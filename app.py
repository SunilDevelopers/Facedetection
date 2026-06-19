from flask import Flask, request, jsonify
import cv2
import base64
import numpy as np
import face_recognition
import os
import sqlite3
from datetime import datetime
import json
# =========================================
# Flask App Initialize
# =========================================
app = Flask(__name__)

# ================= FOLDER =================
FACE_DIR = "faces"
os.makedirs(FACE_DIR, exist_ok=True)

# ================= DB =================
def get_db():
    return sqlite3.connect("attendance.db", check_same_thread=False)

def init_db():
    conn = get_db()

    # USERS TABLE
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        emp_id TEXT,
        mobile TEXT,
        email TEXT,
        password TEXT,
        address TEXT,
        user_type TEXT,
        skill TEXT,
        police_verification TEXT,
        id_card TEXT,
        image TEXT
    )
    """)

    # ATTENDANCE TABLE
    conn.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        emp_id TEXT,
        name TEXT,
        user_type TEXT,
        train_no TEXT,
        direction TEXT,
        enroute TEXT,
        punch_type TEXT,
        date TEXT,
        time TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ================= LOAD FACES =================
known_encodings = []
known_names = []

def load_faces():
    known_encodings.clear()
    known_names.clear()

    for file in os.listdir(FACE_DIR):
        if file.endswith(".npy"):
            try:
                enc = np.load(os.path.join(FACE_DIR, file))

                emp_id = file.replace(".npy", "")   #  FIX

                known_encodings.append(enc)
                known_names.append(emp_id)         #  NOW emp_id store ho raha

            except:
                pass

load_faces()

# ================= DECODE IMAGE =================
def decode_image(image_data):
    try:
        if "," in image_data:
            image_data = image_data.split(",")[1]

        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    except:
        return None

# ================= REGISTER =================
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json(force=True)

        name = data.get("name")
        emp_id = data.get("emp_id")   #  IMPORTANT
        img_str = data.get("image")

        if not name or not img_str or not emp_id:
            return jsonify({"error": "missing data"}), 400

        frame = decode_image(img_str)
        if frame is None:
            return jsonify({"error": "decode failed"}), 400

        rgb = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)

        encodings = face_recognition.face_encodings(rgb)
        if len(encodings) == 0:
            return jsonify({"error": "no face detected"}), 400

        encoding = encodings[0]

        #  IMAGE SAVE
        image_path = os.path.abspath(f"{FACE_DIR}/{emp_id}.jpg")
        cv2.imwrite(image_path, frame)

        # FACE SAVE BY EMP_ID (MOST IMPORTANT)
        np.save(os.path.join(FACE_DIR, f"{emp_id}.npy"), encoding)

        #  DB SAVE
        conn = get_db()
        conn.execute("""
        INSERT INTO users (
            name, emp_id, mobile, email, password,
            address, user_type, skill,
            police_verification, id_card, image
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            name,
            emp_id,
            data.get("mobile"),
            data.get("email"),
            data.get("password"),
            data.get("address"),
            data.get("user_type"),
            data.get("skill"),
            data.get("police_verification"),
            data.get("id_card"),
            image_path
        ))

        conn.commit()
        conn.close()

        load_faces()

        return jsonify({"status": "registered", "name": name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= USERS =================
@app.route("/users", methods=["GET"])
def users():
    try:
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        conn.close()

        print("TOTAL USERS:", len(rows))

        data = []

        for r in rows:
            image_base64 = ""

            image_path = r[11]

            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as img:
                    image_base64 = base64.b64encode(img.read()).decode("utf-8")

            data.append({
                "name": r[1],
                "emp_id": r[2],
                "mobile": r[3],
                "email": r[4],
                "password": r[5],
                "address": r[6],
                "user_type": r[7],
                "skill": r[8],
                "police_verification": r[9],
                "id_card": r[10],
                "image": image_base64
            })

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= MARKATTENDANCE =================

@app.route("/markattendance", methods=["POST"])
def markattendance():
    conn = None
    try:
        data = request.get_json(force=True)
        frame = decode_image(data.get("image"))

        if frame is None:
            return jsonify({"error": "decode failed"}), 400

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        if not encodings:
            return jsonify({"status": "no_face", "name": "No Face"})

        if len(known_encodings) == 0:
            return jsonify({"error": "no registered faces"}), 400

        distances = face_recognition.face_distance(known_encodings, encodings[0])

        best_index = np.argmin(distances)

        name = "Unknown"
        emp_id = None
        user_type = None
        status = "not_matched"

        if distances[best_index] < 0.5:

            emp_id = known_names[best_index]
            status = "matched"

            conn = get_db()
            cursor = conn.cursor()

            # ================= USER FETCH =================
            cursor.execute("""
                SELECT name, user_type
                FROM users
                WHERE emp_id = ?
                LIMIT 1
            """, (emp_id,))

            row = cursor.fetchone()

            if row:
                name, user_type = row

            punch_type = data.get("punch_type")
            enroute = data.get("enroute")

            # ================= LAST ENTRY CHECK =================
            cursor.execute("""
                SELECT punch_type
                FROM attendance
                WHERE emp_id = ?
                ORDER BY id DESC
                LIMIT 1
            """, (emp_id,))

            last = cursor.fetchone()

            # ================= RULES =================

            #  End pe OUT only
            if enroute == "End" and punch_type == "IN":
                return jsonify({
                    "status": "error",
                    "name": "End pe OUT hi allowed hai"
                })

            #  Start/Enroute pe OUT not allowed
            if enroute in ["Start", "En route"] and punch_type == "OUT":
                return jsonify({
                    "status": "error",
                    "name": "OUT Not allowed"
                })

            #  OUT without IN
            if punch_type == "OUT":
                if not last or last[0] != "IN":
                    return jsonify({
                        "status": "error",
                      "name": "Please mark IN before OUT"
                    })

            # ================= INSERT =================
            now = datetime.now()

            conn.execute("""
                INSERT INTO attendance 
                (emp_id, name, user_type, train_no, direction, enroute, punch_type, date, time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                emp_id,
                name,
                user_type,
                data.get("train_no"),
                data.get("direction"),
                enroute,
                punch_type,
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S")
            ))

            conn.commit()

        return jsonify({
            "status": status,
            "emp_id": emp_id,
            "name": name,
            "user_type": user_type
        })

    except Exception as e:
        print("FATAL ERROR:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        if conn:
            conn.close()


# ================= REPORT =================

@app.route("/report", methods=["GET"])
def report():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT 
        a.id,
        a.name,
        a.date,
        a.time,
        a.emp_id,
        a.user_type,
        a.train_no,
        a.direction,
        a.enroute,
        a.punch_type,
        u.image
    FROM attendance a
    LEFT JOIN users u ON a.emp_id = u.emp_id
    ORDER BY a.id DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    result = []

    for r in rows:
        image_base64 = ""
        image_data = r[10]   #  rename (safe)

        # ================= FIX =================
        if image_data and isinstance(image_data, str):

            # agar already base64 hai to direct use karo
            if len(image_data) > 200:   # base64 image check
                image_base64 = image_data

            else:
                # agar file path ho tab hi read karo
                image_path = os.path.abspath(image_data)

                if os.path.exists(image_path):
                    try:
                        with open(image_path, "rb") as img:
                            image_base64 = base64.b64encode(img.read()).decode("utf-8")
                    except Exception as e:
                        print("IMAGE ERROR:", e)

        result.append({
            "id": r[0],
            "name": r[1],
            "date": r[2],
            "time": r[3],
            "emp_id": r[4],
            "user_type": r[5],
            "train_no": r[6],
            "direction": r[7],
            "enroute": r[8],
            "punch_type": r[9],
            "image": image_base64
        })

    return jsonify(result)

# ================= REPORT DELETE =================

@app.route("/delete_report", methods=["POST"])
def delete_report():
    data = request.get_json()

    record_id = data.get("id")   #  ID use karo

    conn = get_db()
    conn.execute(
        "DELETE FROM attendance WHERE id=?",
        (record_id,)
    )
    conn.commit()
    conn.close()

    return jsonify({"status": "deleted"})

# ================= DELETE =================

@app.route("/delete_user", methods=["POST"])
def delete_user():
    try:
        name = request.json.get("name")

        if not name:
            return jsonify({"error": "name required"}), 400

        conn = get_db()
        cursor = conn.cursor()

        # User details fetch
        cursor.execute("""
            SELECT emp_id, image
            FROM users
            WHERE TRIM(name)=TRIM(?)
            LIMIT 1
        """, (name,))

        user = cursor.fetchone()

        if not user:
            conn.close()
            return jsonify({"error": "user not found"}), 404

        emp_id = user[0]
        image_path = user[1]

        # Delete DB Records
        cursor.execute(
            "DELETE FROM users WHERE emp_id=?",
            (emp_id,)
        )

        cursor.execute(
            "DELETE FROM attendance WHERE emp_id=?",
            (emp_id,)
        )

        conn.commit()
        conn.close()

        # Delete Face Encoding File
        face_file = os.path.join(
            FACE_DIR,
            f"{emp_id}.npy"
        )

        if os.path.exists(face_file):
            os.remove(face_file)

        # Delete Image File
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

        # Reload Face List
        load_faces()

        return jsonify({
            "status": "deleted",
            "emp_id": emp_id,
            "name": name
        })

    except Exception as e:
        print("DELETE ERROR:", str(e))
        return jsonify({
            "error": str(e)
        }), 500
    
 # =================UPDATE-USER================

@app.route("/update_user", methods=["POST"])
def update_user():
    try:
        data = request.get_json(force=True)

        old_name = data.get("old_name")
        new_name = data.get("new_name")
        new_emp_id = data.get("emp_id")
        image = data.get("image")

        if not old_name:
            return jsonify({"error": "old_name required"}), 400

        conn = get_db()
        cursor = conn.cursor()

        # ================= FIND OLD USER =================
        cursor.execute("""
            SELECT emp_id,image
            FROM users
            WHERE TRIM(name)=TRIM(?)
            LIMIT 1
        """, (old_name,))

        user = cursor.fetchone()

        if not user:
            conn.close()
            return jsonify({"error": "user not found"}), 404

        old_emp_id = user[0]
        old_image_path = user[1]

        frame = None
        encoding = None
        image_path = old_image_path

        # ================= IMAGE PROCESS =================
        if image and image.strip() != "":
            frame = decode_image(image)

            if frame is not None:

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                encodings = face_recognition.face_encodings(rgb)

                if len(encodings) > 0:
                    encoding = encodings[0]

                image_path = os.path.abspath(
                    os.path.join(
                        FACE_DIR,
                        f"{new_emp_id}.jpg"
                    )
                )

                cv2.imwrite(image_path, frame)

        # ================= USERS UPDATE =================
        cursor.execute("""
            UPDATE users SET
                name=?,
                emp_id=?,
                mobile=?,
                email=?,
                password=?,
                address=?,
                user_type=?,
                skill=?,
                police_verification=?,
                id_card=?,
                image=?
            WHERE emp_id=?
        """, (
            new_name,
            new_emp_id,
            data.get("mobile"),
            data.get("email"),
            data.get("password"),
            data.get("address"),
            data.get("user_type"),
            data.get("skill"),
            data.get("police_verification"),
            data.get("id_card"),
            image_path,
            old_emp_id
        ))

        # ================= ATTENDANCE UPDATE =================
        cursor.execute("""
            UPDATE attendance
            SET name=?,
                emp_id=?,
                user_type=?
            WHERE emp_id=?
        """, (
            new_name,
            new_emp_id,
            data.get("user_type"),
            old_emp_id
        ))

        conn.commit()
        conn.close()

        # ================= FACE FILE UPDATE =================
        if encoding is not None:

            old_face_file = os.path.join(
                FACE_DIR,
                f"{old_emp_id}.npy"
            )

            if os.path.exists(old_face_file):
                os.remove(old_face_file)

            np.save(
                os.path.join(
                    FACE_DIR,
                    f"{new_emp_id}.npy"
                ),
                encoding
            )

        # ================= OLD IMAGE DELETE =================
        if (
            old_image_path
            and old_image_path != image_path
            and os.path.exists(old_image_path)
        ):
            try:
                os.remove(old_image_path)
            except:
                pass

        # ================= RELOAD FACE LIST =================
        load_faces()

        return jsonify({
            "status": "updated",
            "old_emp_id": old_emp_id,
            "new_emp_id": new_emp_id,
            "name": new_name
        })

    except Exception as e:
        print("UPDATE ERROR:", str(e))
        return jsonify({
            "error": str(e)
        }), 500


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# # ================= RUN =================
# if __name__ == "__main__":
#     app.run(host="65.254.80.35", port=5000, debug=True)