import os
import mysql.connector
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import onnxruntime as ort
import cv2
import threading
import time
import ttkbootstrap as ttkb


class TumorSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("로그인")
        self.root.geometry("300x250")
        self.is_running = False
        self.deeplab_model = None
        self.db_connection = None
        self.logged_in_user_id = None  # 로그인한 사용자의 ID를 저장

        # Set the appearance mode and theme for ttkbootstrap
        root.style = ttkb.Style()
        root.style.theme_use("litera")  # or any theme of your choice

        # 로그인 창 UI 구성
        ttkb.Label(root, text="이메일:").pack(pady=(10, 5))
        self.email_entry = ttkb.Entry(root)
        self.email_entry.pack(pady=5)

        ttkb.Label(root, text="비밀번호:").pack(pady=5)
        self.password_entry = ttkb.Entry(root, show='*')
        self.password_entry.pack(pady=5)

        ttkb.Button(root, text="로그인", command=self.login).pack(pady=10)
        ttkb.Button(root, text="회원가입", command=self.open_signup_window).pack()

    def connect_to_database(self):
        if not self.db_connection:
            self.db_connection = mysql.connector.connect(
                host="127.0.0.1",
                user="root",
                password="0000",
                database="mydb"
            )
        return self.db_connection

    def login(self):
        email = self.email_entry.get()
        password = self.password_entry.get()
        
        db = self.connect_to_database()
        cursor = db.cursor()
        cursor.execute("SELECT id FROM users WHERE email=%s AND password=%s", (email, password))
        user = cursor.fetchone()

        if user:
            self.logged_in_user_id = user[0]  # 로그인한 사용자의 ID 저장
            cursor.execute("UPDATE users SET last_login = NOW() WHERE email = %s", (email,))
            db.commit()
            
            messagebox.showinfo("로그인 성공", "로그인되었습니다.")
            self.open_segmentation_app()
        else:
            messagebox.showerror("로그인 실패", "이메일 또는 비밀번호가 잘못되었습니다.")

        cursor.close()
    
    def open_signup_window(self):
        # Signup 창 UI
        signup_window = tk.Toplevel(self.root)
        signup_window.title("회원가입")
        signup_window.geometry("300x300")

        # 이메일 입력 필드
        signup_email_label = ttkb.Label(signup_window, text="이메일:")
        signup_email_label.pack()
        signup_email_entry = ttkb.Entry(signup_window)
        signup_email_entry.pack()

        # 사용자 이름 (닉네임) 입력 필드
        signup_username_label = ttkb.Label(signup_window, text="사용자 이름 (닉네임):")
        signup_username_label.pack()
        signup_username_entry = ttkb.Entry(signup_window)
        signup_username_entry.pack()

        # 비밀번호 입력 필드
        signup_password_label = ttkb.Label(signup_window, text="비밀번호:")
        signup_password_label.pack()
        signup_password_entry = ttkb.Entry(signup_window, show='*')
        signup_password_entry.pack()

        # 전화번호 입력 필드
        signup_phone_label = ttkb.Label(signup_window, text="전화번호 (선택 사항):")
        signup_phone_label.pack()
        signup_phone_entry = ttkb.Entry(signup_window)
        signup_phone_entry.pack()

        # 회원가입 버튼
        def signup():
            email = signup_email_entry.get()
            username = signup_username_entry.get()
            password = signup_password_entry.get()
            phone = signup_phone_entry.get() or ''  # 전화번호가 빈칸이면 빈 문자열로 처리

            db = self.connect_to_database()
            cursor = db.cursor()
            try:
                cursor.execute(
                    "INSERT INTO users (email, username, password, phone) VALUES (%s, %s, %s, %s)",
                    (email, username, password, phone)
                )
                db.commit()
                messagebox.showinfo("회원가입 성공", "계정이 생성되었습니다.")
                signup_window.destroy()  # 회원가입 창 닫기
                self.root.deiconify()  # 로그인 창 다시 표시
            except mysql.connector.Error as err:
                messagebox.showerror("회원가입 실패", str(err))
            finally:
                cursor.close()

        signup_button = ttkb.Button(signup_window, text="회원가입", command=signup)
        signup_button.pack(pady=10)

    def open_segmentation_app(self):
        self.root.withdraw()
        segment_window = tk.Toplevel(self.root)
        segment_window.title("유방 종양 분할 프로그램")
        segment_window.geometry("700x700")
        
        self.deeplab_model = self.load_onnx_model(r"C:\Users\ykm\CAPST2\model\deeplabv3.onnx")
        
        self.image_label = tk.Label(segment_window)
        self.image_label.pack()

        self.progress_bar = ttkb.Progressbar(segment_window, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)
        
        self.progress_label = ttkb.Label(segment_window, text="0%")
        self.progress_label.pack()

        self.result_label = ttkb.Label(segment_window, text="결과: ")
        self.result_label.pack()

        ttkb.Button(segment_window, text="이미지 업로드", command=self.open_file).pack(pady=5)
        ttkb.Button(segment_window, text="분할 실행", command=self.segment_image).pack(pady=5)
        ttkb.Button(segment_window, text="중지", command=self.stop_segmentation).pack(pady=5)
        ttkb.Button(segment_window, text="로컬에 저장", command=self.save_images_to_local).pack(pady=5)

    def preprocess_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("이미지를 읽을 수 없습니다. 경로를 확인하세요:", file_path)
            return None
        
        img = cv2.resize(img, (256, 256))
        return img[np.newaxis, np.newaxis, :, :].astype(np.float32)

    def load_onnx_model(self, model_path):
        return ort.InferenceSession(model_path)

    def segment_image(self):
        self.is_running = True
        self.result_label.config(text="분할 중...")
        self.progress_bar["value"] = 0
        self.progress_label.config(text="0%")

        def run_segmentation():
            if hasattr(self.root, 'img_path'):
                input_data = self.preprocess_image(self.root.img_path)
                if input_data is None:
                    self.result_label.config(text="이미지를 불러올 수 없습니다.")
                    return

                for i in range(100):
                    if not self.is_running:
                        self.result_label.config(text="분할 중지됨")
                        return
                    self.progress_bar["value"] = i + 1
                    self.progress_label.config(text=f"{i + 1}%")
                    time.sleep(0.05)
                    self.root.update_idletasks()

                input_name = self.deeplab_model.get_inputs()[0].name
                outputs = self.deeplab_model.run(None, {input_name: input_data})
                
                segmentation_mask = outputs[0][0]
                segmentation_mask = np.argmax(segmentation_mask, axis=0)
                
                output_image = np.zeros((256, 256), dtype=np.uint8)
                output_image[segmentation_mask == 1] = 255
                
                self.save_images_to_db(self.root.img_path, output_image)

                mask_image = Image.fromarray(output_image)
                mask_tk = ImageTk.PhotoImage(mask_image.resize((256, 256)))
                self.image_label.config(image=mask_tk)
                self.image_label.image = mask_tk
                
                self.result_label.config(text="분할 결과: 완료")
                self.progress_bar["value"] = 100
                self.progress_label.config(text="100%")

        threading.Thread(target=run_segmentation).start()

    def save_images_to_db(self, original_image_path, segmented_image):
        if self.logged_in_user_id is None:
            messagebox.showerror("오류", "로그인 후 이미지를 저장하세요.")
            return

        db = self.connect_to_database()
        cursor = db.cursor()

        with open(original_image_path, "rb") as file:
            original_image_blob = file.read()

        _, segmented_image_encoded = cv2.imencode('.png', segmented_image)
        segmented_image_blob = segmented_image_encoded.tobytes()

        sql = """
        INSERT INTO images (user_id, original_image, segmented_image)
        VALUES (%s, %s, %s)
        """
        cursor.execute(sql, (self.logged_in_user_id, original_image_blob, segmented_image_blob))
        db.commit()
    def save_images_to_local(self):
        if self.logged_in_user_id is None:
            messagebox.showerror("오류", "로그인 후 이미지를 저장하세요.")
            return
        
        # 폴더 선택 대화 상자 열기
        output_directory = filedialog.askdirectory(title="저장할 폴더 선택")
        if not output_directory:
            messagebox.showinfo("알림", "폴더를 선택하지 않았습니다.")
            return

        # 사용자별 폴더 생성
        user_folder = os.path.join(output_directory, f"user_{self.logged_in_user_id}_images")
        os.makedirs(user_folder, exist_ok=True)

        # 데이터베이스에서 사용자 이미지 조회
        db = self.connect_to_database()
        cursor = db.cursor()
        try:
            cursor.execute("SELECT image_id, original_image, segmented_image FROM images WHERE user_id = %s", (self.logged_in_user_id,))
            images = cursor.fetchall()

            # 이미지 저장
            for image_id, original_image_blob, segmented_image_blob in images:
                original_path = os.path.join(user_folder, f"original_image_{image_id}.png")
                segmented_path = os.path.join(user_folder, f"segmented_image_{image_id}.png")

                with open(original_path, "wb") as original_file:
                    original_file.write(original_image_blob)

                with open(segmented_path, "wb") as segmented_file:
                    segmented_file.write(segmented_image_blob)

            messagebox.showinfo("저장 완료", f"모든 이미지가 {user_folder}에 저장되었습니다.")
        except mysql.connector.Error as err:
            messagebox.showerror("오류", f"이미지 저장 중 오류가 발생했습니다: {str(err)}")
        finally:
            cursor.close()
        

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.root.img_path = file_path
            image = Image.open(file_path)
            image.thumbnail((300, 300))  # 이미지 크기 조정
            image_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=image_tk)
            self.image_label.image = image_tk

    def stop_segmentation(self):
        self.is_running = False
        self.result_label.config(text="분할 중지됨")
        self.progress_bar["value"] = 0
        self.progress_label.config(text="0%")


if __name__ == "__main__":
    root = tk.Tk()
    app = TumorSegmentationApp(root)
    root.mainloop()
