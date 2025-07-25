import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import onnxruntime as ort
import cv2
import threading
import time

def preprocess_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("이미지 파일 경로:", file_path)
        print("이미지를 읽을 수 없습니다. 경로를 확인하세요.")
        return None
    
    img = cv2.resize(img, (256, 256))
    img_array = img[np.newaxis, np.newaxis, :, :].astype(np.float32)
    return img_array

def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

# ONNX 모델 로드
deeplab_model = load_onnx_model(r"C:\Users\ykm\CAPST2\model\deeplabv3.onnx")

# 분석 중지를 위한 플래그
is_running = False

def show_original_image():
    if hasattr(root, 'img_path'):
        img = Image.open(root.img_path).resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

def save_segmented_image(output_image):
    save_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                               filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if save_path:
        output_image.save(save_path)

def segment_image():
    global is_running
    is_running = True
    result_label.config(text="분할 중...")
    progress_bar["value"] = 0
    progress_label.config(text="0%")

    def run_segmentation():
        if hasattr(root, 'img_path'):
            input_data = preprocess_image(root.img_path)
            if input_data is None:
                progress_bar["value"] = 0
                progress_label.config(text="0%")
                result_label.config(text="이미지를 불러올 수 없습니다.")
                return
            
            for i in range(100):
                if not is_running:
                    result_label.config(text="분할 중지됨")
                    return

                progress_bar["value"] = i + 1
                progress_label.config(text=f"{i + 1}%")
                time.sleep(0.05)
                root.update_idletasks()

            input_name = deeplab_model.get_inputs()[0].name
            outputs = deeplab_model.run(None, {input_name: input_data})
            
            segmentation_mask = outputs[0][0]
            segmentation_mask = np.argmax(segmentation_mask, axis=0)
            
            output_image = np.zeros((256, 256), dtype=np.uint8)
            output_image[segmentation_mask == 1] = 255

            mask_image = Image.fromarray(output_image)
            mask_image = mask_image.resize((256, 256))
            mask_tk = ImageTk.PhotoImage(mask_image)
            image_label.config(image=mask_tk)
            image_label.image = mask_tk
            
            result_label.config(text="분할 결과: 완료")
            progress_bar["value"] = 100
            progress_label.config(text="100%")

            # 원본 이미지 다시 보기 버튼 생성
            show_original_button = tk.Button(root, text="원본 이미지 다시 보기", command=show_original_image)
            show_original_button.pack()

            # 분할된 이미지 저장 버튼 생성
            save_button = tk.Button(root, text="분할 이미지 저장", command=lambda: save_segmented_image(mask_image))
            save_button.pack()

    threading.Thread(target=run_segmentation).start()

def open_file():
    global is_running
    is_running = False
    progress_bar["value"] = 0
    progress_label.config(text="0%")
    result_label.config(text="결과: ")
    file_path = filedialog.askopenfilename()
    if file_path:
        root.img_path = file_path
        img = Image.open(file_path).resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

def stop_segmentation():
    global is_running
    is_running = False

# Tkinter 윈도우 설정
root = tk.Tk()
root.title("유방 종양 분할 프로그램")
root.geometry("500x600")

# 파일 선택 버튼
open_button = tk.Button(root, text="이미지 업로드", command=open_file)
open_button.pack()

# 이미지를 표시할 레이블
image_label = tk.Label(root)
image_label.pack()

# 분할 실행 버튼
segment_button = tk.Button(root, text="분할 실행", command=segment_image)
segment_button.pack()

# 진행 표시줄
progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=10)

# 진행률 텍스트 레이블
progress_label = tk.Label(root, text="0%")
progress_label.pack()

# 중지 버튼
stop_button = tk.Button(root, text="중지", command=stop_segmentation)
stop_button.pack()

# 결과 표시 레이블
result_label = tk.Label(root, text="결과: ")
result_label.pack()

root.mainloop()
