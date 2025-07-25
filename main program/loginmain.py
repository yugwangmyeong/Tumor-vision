import tkinter as tk
from tkinter import messagebox
import mysql.connector

# MySQL 데이터베이스 연결 설정
def create_connection():
    return mysql.connector.connect(
        host="127.0.0.1",        # 데이터베이스 서버 주소
        user="root",    # MySQL 사용자 이름
        password="0000", # MySQL 비밀번호
        database="mydb"  # 데이터베이스 이름
    )

def login():
    username = username_entry.get()
    password = password_entry.get()

    # 데이터베이스에 연결
    connection = create_connection()
    cursor = connection.cursor()

    # 사용자 인증 쿼리
    query = "SELECT * FROM users WHERE username = %s AND password = %s"
    cursor.execute(query, (username, password))
    result = cursor.fetchone()  # 첫 번째 결과를 가져옴

    if result:  # 사용자 인증 성공
        messagebox.showinfo("로그인 성공", "환영합니다!")
        root.destroy()  # 로그인 창 닫기
        # 여기서 메인 애플리케이션 코드 호출

    else:  # 사용자 인증 실패
        messagebox.showerror("로그인 실패", "잘못된 사용자 이름 또는 비밀번호입니다.")

    # 연결 닫기
    cursor.close()
    connection.close()

def register():
    username = reg_username_entry.get()
    password = reg_password_entry.get()

    # 데이터베이스에 연결
    connection = create_connection()
    cursor = connection.cursor()

    # 사용자 등록 쿼리
    try:
        query = "INSERT INTO users (username, password) VALUES (%s, %s)"
        cursor.execute(query, (username, password))
        connection.commit()  # 변경 사항 저장
        messagebox.showinfo("회원가입 성공", "회원가입이 완료되었습니다!")
        registration_window.destroy()  # 회원가입 창 닫기
    except mysql.connector.Error as err:
        messagebox.showerror("회원가입 실패", f"오류 발생: {err}")
    finally:
        # 연결 닫기
        cursor.close()
        connection.close()

def open_registration_window():
    global registration_window, reg_username_entry, reg_password_entry

    registration_window = tk.Toplevel(root)
    registration_window.title("회원가입")
    registration_window.geometry("300x200")

    reg_username_label = tk.Label(registration_window, text="사용자 이름:")
    reg_username_label.pack(pady=5)
    reg_username_entry = tk.Entry(registration_window)
    reg_username_entry.pack(pady=5)

    reg_password_label = tk.Label(registration_window, text="비밀번호:")
    reg_password_label.pack(pady=5)
    reg_password_entry = tk.Entry(registration_window, show="*")
    reg_password_entry.pack(pady=5)

    register_button = tk.Button(registration_window, text="회원가입", command=register)
    register_button.pack(pady=20)

# Tkinter 윈도우 설정
root = tk.Tk()
root.title("로그인")
root.geometry("300x250")

# 사용자 이름 레이블 및 입력 필드
username_label = tk.Label(root, text="사용자 이름:")
username_label.pack(pady=5)
username_entry = tk.Entry(root)
username_entry.pack(pady=5)

# 비밀번호 레이블 및 입력 필드
password_label = tk.Label(root, text="비밀번호:")
password_label.pack(pady=5)
password_entry = tk.Entry(root, show="*")  # 비밀번호 입력은 '*'로 표시
password_entry.pack(pady=5)

# 로그인 버튼
login_button = tk.Button(root, text="로그인", command=login)
login_button.pack(pady=20)

# 회원가입 버튼
register_button = tk.Button(root, text="회원가입", command=open_registration_window)
register_button.pack(pady=5)

root.mainloop()
