import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from recognizer import train, recognize

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition using Eigenface")
        self.root.geometry("850x600")
        self.root.configure(bg="#EAF6FF")

        self.dataset_path = ""
        self.test_image_path = ""
        self.eigenfaces = None
        self.projections = None
        self.mean_face = None
        self.filenames = None

        self.test_photo = None
        self.result_photo = None

        self.setup_ui()

    def setup_ui(self):
        title = tk.Label(self.root, text="Face Recognition System", font=("Segoe UI", 24, "bold"), bg="#EAF6FF", fg="#005792")
        title.pack(pady=20)

        btn_frame = tk.Frame(self.root, bg="#EAF6FF")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="📂 Insert Dataset", command=self.load_dataset, width=20, bg="#007ACC", fg="white", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="🖼️ Insert Test Image", command=self.load_test_image, width=20, bg="#007ACC", fg="white", font=("Segoe UI", 10, "bold")).grid(row=0, column=1, padx=10)
        tk.Button(btn_frame, text="🔍 Match Face", command=self.match, width=20, bg="#28A745", fg="white", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, padx=10)

        image_frame = tk.Frame(self.root, bg="#EAF6FF")
        image_frame.pack(pady=20)

        tk.Label(image_frame, text="Test Image", font=("Segoe UI", 12), bg="#EAF6FF").grid(row=0, column=0)
        tk.Label(image_frame, text="Closest Match", font=("Segoe UI", 12), bg="#EAF6FF").grid(row=0, column=1)

        self.test_image_panel = tk.Label(image_frame, bg="#D6E6F2", width=200, height=200)
        self.test_image_panel.grid(row=1, column=0, padx=40, pady=10)

        self.result_image_panel = tk.Label(image_frame, bg="#D6E6F2", width=200, height=200)
        self.result_image_panel.grid(row=1, column=1, padx=40, pady=10)

        self.result_text = tk.Label(self.root, text="Result: None", font=("Segoe UI", 12), bg="#EAF6FF", fg="#333333")
        self.result_text.pack(pady=10)

    def load_dataset(self):
        path = filedialog.askdirectory()
        if path:
            try:
                self.dataset_path = path
                self.eigenfaces, self.projections, self.mean_face, self.filenames = train(path)
                messagebox.showinfo("Dataset Loaded", "✅ Dataset berhasil dimuat!")
            except Exception as e:
                messagebox.showerror("Error", f"❌ Gagal memuat dataset:\n{e}")

    def load_test_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            self.test_image_path = path
            try:
                img = Image.open(path).resize((200, 200))
                self.test_photo = ImageTk.PhotoImage(img)
                self.test_image_panel.config(image=self.test_photo)
                self.test_image_panel.image = self.test_photo
            except:
                messagebox.showerror("Error", "❌ Gambar tidak bisa ditampilkan.")

    def match(self):
        if not self.dataset_path or not self.test_image_path:
            messagebox.showwarning("Peringatan", "⚠ Harap pilih dataset dan gambar terlebih dahulu.")
            return

        try:
            match_name, distance = recognize(
                self.test_image_path,
                self.eigenfaces,
                self.projections,
                self.mean_face,
                self.filenames,
                threshold=999999999  # selalu tampilkan closest match
            )

            matched_img_path = os.path.join(self.dataset_path, match_name) if match_name else None

            if matched_img_path and os.path.exists(matched_img_path):
                img = Image.open(matched_img_path).resize((200, 200))
                self.result_photo = ImageTk.PhotoImage(img)
                self.result_image_panel.config(image=self.result_photo)
                self.result_image_panel.image = self.result_photo
                self.result_text.config(text=f"✅ Closest match: {match_name}\nDistance: {distance:.2f}")
            else:
                self.result_image_panel.config(image='')
                self.result_text.config(text=f"❌ Tidak ditemukan wajah. Closest tetap ditampilkan:\n{match_name}\nDistance: {distance:.2f}")

        except Exception as e:
            print(f"[ERROR] {e}")
            messagebox.showerror("Error", f"❌ Terjadi kesalahan saat mencocokkan:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
