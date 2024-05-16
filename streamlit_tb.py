import av
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import time
from io import BytesIO


# Konfigurasi awal halaman Streamlit
st.set_page_config(page_title="Lungs.AI", page_icon="icon_tb.ico")

st.markdown(
    """
    <style>
        .big-font {
            font-size:18px !important;
        }
    </style>
    
    """, 
    unsafe_allow_html=True
)
# Fungsi untuk menampilkan animasi loading saat memuat halaman
def show_page_loading():
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown("<h1 style='text-align: center;'>Memuat Halaman...</h1>", unsafe_allow_html=True)
        # Animasi loading
        st.markdown("""
            <style>
            .loader-container {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }

            .loader {
                border: 5px solid #f3f3f3;
                border-radius: 50%;
                border-top: 5px solid #F14848;
                width: 40px;
                height: 40px;
                -webkit-animation: spin 2s linear infinite; /* Safari */
                animation: spin 2s linear infinite;
            }

            @-webkit-keyframes spin {
                0% { -webkit-transform: rotate(0deg); }
                100% { -webkit-transform: rotate(360deg); }
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            </style>
            <div class="loader-container">
                <div class="loader"></div>
            </div>
            """, unsafe_allow_html=True)
    
        time.sleep(2)  # Simulasi delay loading halaman
    loading_placeholder.empty()  # Menghapus pesan loading setelah selesai

# Fungsi untuk menampilkan animasi loading kustom
def show_custom_loading():
    st.markdown("""
        <style>
        .loader {
            border: 5px solid #f3f3f3;
            border-radius: 50%;
            border-top: 5px solid ##F14848;
            width: 40px;
            height: 40px;
            -webkit-animation: spin 2s linear infinite; /* Safari */
            animation: spin 2s linear infinite;
        }

        /* Safari */
        @-webkit-keyframes spin {
            0% { -webkit-transform: rotate(0deg); }
            100% { -webkit-transform: rotate(360deg); }
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        <div class="loader"></div>
        """, unsafe_allow_html=True)
    
# Fungsi untuk menampilkan pesan loading dengan spinner
def show_loading_message(message):
    with st.empty():
        st.markdown(f"<h3 style='text-align: center;'>{message}</h3>", unsafe_allow_html=True)
        # Create a progress bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.04)  # Set sleep time so total duration is 2 seconds (0.02 * 100)
            progress_bar.progress(percent_complete + 1)
        st.empty()   


# Memuat model AI
model_tb = load_model('C:\\Users\\user\\appdata\\keras_model_(new).h5')
model_covid = load_model('C:\\Users\\user\\appdata\\model_covid.h5')
model_pneumonia = load_model('C:\\Users\\user\\appdata\\keras_model_pneumonia.h5')




# Kelas untuk transformasi video pada pemeriksaan realtwime
class VideoTransformer(VideoTransformerBase):


    def recv(self, frame):
        # Logika untuk mengolah frame video dan melakukan prediksi
        class_names = [line.strip() for line in open("C://Users//user//AppData//labels(new).txt", "r").readlines()]
        image = frame.to_ndarray(format="bgr24")
        # Resize dan preprosses gambar
        resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        image_array = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
        image_array = (image_array / 127.5) - 1

        # membuat prediksi
        prediction = model_covid.predict(image_array)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        text = f"{class_name}: {confidence_score:.2f}"
        font_scale = 0.6
        font_thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_offset_x = 10
        text_offset_y = resized_image.shape[0] - 20
        box_coords = ((text_offset_x, text_offset_y + 5), (text_offset_x + text_width + 5, text_offset_y - text_height - 5))
        cv2.rectangle(resized_image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)

        # menampilkan prediksi pada gambar
        cv2.putText(resized_image, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(resized_image, format="bgr24")

# Fungsi untuk melakukan prediksi pada gambar yang diunggah
def prediksi_gambar_tb(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    prediksi_tb = model_tb.predict(img_tensor)
    kelas_label = ['Normal', 'Tuberkulosis']
    indeks_kelas = np.argmax(prediksi_tb[0])
    label_kelas = kelas_label[indeks_kelas]
    skor_kepercayaan = float(prediksi_tb[0][indeks_kelas])

    hasil = {
        'label_kelas': label_kelas,
        'skor_kepercayaan': skor_kepercayaan
    }

    return hasil

def prediksi_gambar_covid(file_path):
    # Membaca gambar menggunakan OpenCV
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))  # Sesuaikan ukuran gambar dengan input model
    img_tensor = image.img_to_array(img)  # Mengubah gambar menjadi tensor
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Menambahkan dimensi batch
    img_tensor /= 255.  # Normalisasi gambar

    # Membuat prediksi menggunakan model
    prediksi_covid = model_covid.predict(img_tensor)

    # Mengambil indeks dengan nilai prediksi tertinggi
    indeks_kelas = np.argmax(prediksi_covid[0])
    kelas_label = ['normal', 'covid']  # Sesuaikan dengan label kelas yang Anda miliki
    label_kelas = kelas_label[indeks_kelas]
    skor_kepercayaan = float(prediksi_covid[0][indeks_kelas])

    hasil = {
        'label_kelas': label_kelas,
        'skor_kepercayaan': skor_kepercayaan
    }

    return hasil

def prediksi_gambar_pneumonia(file_path):
    # Membaca gambar menggunakan OpenCV
    img = cv2.imread(file_path)
    img = cv2.resize(img, (224, 224))  # Sesuaikan ukuran gambar dengan input model
    img_tensor = image.img_to_array(img)  # Mengubah gambar menjadi tensor
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Menambahkan dimensi batch
    img_tensor /= 255.  # Normalisasi gambar

    # Membuat prediksi menggunakan model
    prediksi_pneumonia = model_pneumonia.predict(img_tensor)

    # Mengambil indeks dengan nilai prediksi tertinggi
    indeks_kelas = np.argmax(prediksi_pneumonia[0])
    kelas_label = ['normal', 'pneumonia']  # Sesuaikan dengan label kelas yang Anda miliki
    label_kelas = kelas_label[indeks_kelas]
    skor_kepercayaan = float(prediksi_pneumonia[0][indeks_kelas])

    hasil = {
        'label_kelas': label_kelas,
        'skor_kepercayaan': skor_kepercayaan
    }

    return hasil
canvas_result = None





# Aplikasi Streamlit

# Navigasi
halaman_terpilih = st.sidebar.radio("Pilih Halaman", ["Beranda", "Halaman Prediksi", "Pemeriksaan Realtime", "Edit Gambar", "Visualisasi Model"], format_func=lambda x: x)
st.sidebar.title("Lungs Check")
if halaman_terpilih == "Beranda":
    show_page_loading()
    # Tampilkan halaman Beranda
    st.header("Selamat datang di Aplikasi Prediksi Tuberkulosis, Covid-19, dan Pneumonia!", divider='rainbow')
    st.markdown(
    """
    <p style='font-size: 23px;'>
        Aplikasi ini memungkinkan Anda untuk mengunggah gambar paru-paru Anda 
        dan mendapatkan prediksi apakah paru-paru Anda normal atau terdeteksi penyakit.
    </p>
    """, 
    unsafe_allow_html=True
)
    st.markdown(
    """
    <p class="big-font">1. Silakan pilih 'Halaman Prediksi' untuk memulai prediksi menggunakan gambar.</p>
    <p class="big-font">2. Silakan pilih 'Pemeriksaan Realtime' untuk memulai prediksi secara Realtime menggunakan kamera.</p>
    <p class="big-font">3. Silakan pilih 'Edit Gambar' untuk Mengedit Gambar Rontgen.</p>
    <p class="big-font">4. Silakan pilih 'Visualisasi Model' untuk mengetahui pemahaman tentang model AI tuberkulosis.</p>
    """, 
    unsafe_allow_html=True
)
elif halaman_terpilih == "Halaman Prediksi":
    show_page_loading()
    # Tampilkan halaman Prediksi
    st.title("Unggah Gambar")
    st.markdown("---")

    # Pilih mode prediksi: TBC atau COVID-19
    mode_prediksi = st.selectbox("Pilih Mode Prediksi", ["TBC", "COVID-19", "Pneumonia"])

    # Unggah gambar melalui Streamlit
    berkas_gambar = st.file_uploader("Silahkan Pilih gambar ", type=["jpg", "jpeg", "png"])

    if berkas_gambar:
        # Tampilkan gambar yang dipilih
        st.image(berkas_gambar, caption="Gambar yang Diunggah", use_column_width=True)
        # Lakukan prediksi saat tombol ditekan
        if st.button("Prediksi"):
             # Menampilkan pesan loading
            show_loading_message("Sedang memproses prediksi, mohon tunggu...")
            # Simpan berkas gambar yang diunggah ke lokasi sementara
            with open("temp_image.jpg", "wb") as f:
                f.write(berkas_gambar.getbuffer())

            # Mode TBC
            if mode_prediksi == "TBC":

            # Lakukan prediksi pada berkas yang disimpan
                hasil_prediksi = prediksi_gambar_tb("temp_image.jpg")
                # Tampilkan hasil prediksi
                st.write(f"Prediksi: {hasil_prediksi['label_kelas']}")
                st.write(f"Skor Kepercayaan: {hasil_prediksi['skor_kepercayaan']:.2%}")
                if hasil_prediksi['label_kelas'] == 'Normal':
                    st.write(
                        "Selamat! Berdasarkan prediksi kami, paru-paru Anda tampaknya dalam keadaan normal. Namun, ingatlah bahwa ini hanya hasil dari model kecerdasan buatan kami. Jika Anda memiliki kekhawatiran kesehatan atau pertanyaan lebih lanjut, sangat disarankan untuk berkonsultasi dengan dokter untuk pemeriksaan yang lebih mendalam."
                    )
                elif hasil_prediksi['label_kelas'] == 'Tuberkulosis':
                    st.write(
                        "Hasil prediksi menunjukkan kemungkinan deteksi TBC pada paru-paru Anda. Namun, perlu diingat bahwa ini hanyalah hasil dari model kecerdasan buatan kami. Kami sarankan Anda untuk segera berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut dan konfirmasi. Jangan ragu untuk mendiskusikan hasil ini bersama profesional kesehatan Anda.")
                
            elif mode_prediksi =="COVID-19":
                hasil_prediksi = prediksi_gambar_covid("temp_image.jpg")
                st.write(f"Prediksi: {hasil_prediksi['label_kelas']}")
                st.write(f"Skor Kepercayaan: {hasil_prediksi['skor_kepercayaan']:.2%}")
                if hasil_prediksi['label_kelas'] == 'normal':
                    st.write(
                        "Selamat! Berdasarkan prediksi kami, paru-paru Anda tampaknya dalam keadaan normal. Namun, ingatlah bahwa ini hanya hasil dari model kecerdasan buatan kami. Jika Anda memiliki kekhawatiran kesehatan atau pertanyaan lebih lanjut, sangat disarankan untuk berkonsultasi dengan dokter untuk pemeriksaan yang lebih mendalam."
                    )
                elif hasil_prediksi['label_kelas'] == 'covid':
                    st.write(
                        "Hasil prediksi menunjukkan kemungkinan deteksi COVID-19 pada paru-paru Anda. Namun, perlu diingat bahwa ini hanyalah hasil dari model kecerdasan buatan kami. Kami sarankan Anda untuk segera berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut dan konfirmasi. Jangan ragu untuk mendiskusikan hasil ini bersama profesional kesehatan Anda.")
            elif mode_prediksi =="Pneumonia":
                hasil_prediksi = prediksi_gambar_pneumonia("temp_image.jpg")
                st.write(f"Prediksi: {hasil_prediksi['label_kelas']}")
                st.write(f"Skor Kepercayaan: {hasil_prediksi['skor_kepercayaan']:.2%}")
                if hasil_prediksi['label_kelas'] == 'normal':
                    st.write(
                        "Selamat! Berdasarkan prediksi kami, paru-paru Anda tampaknya dalam keadaan normal. Namun, ingatlah bahwa ini hanya hasil dari model kecerdasan buatan kami. Jika Anda memiliki kekhawatiran kesehatan atau pertanyaan lebih lanjut, sangat disarankan untuk berkonsultasi dengan dokter untuk pemeriksaan yang lebih mendalam."
                    )
                elif hasil_prediksi['label_kelas'] == 'pneumonia':
                    st.write(
                        "Hasil prediksi menunjukkan kemungkinan deteksi Pneumonia pada paru-paru Anda. Namun, perlu diingat bahwa ini hanyalah hasil dari model kecerdasan buatan kami. Kami sarankan Anda untuk segera berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut dan konfirmasi. Jangan ragu untuk mendiskusikan hasil ini bersama profesional kesehatan Anda.")
                    

# Halaman Pemeriksan Realtime
elif halaman_terpilih == "Pemeriksaan Realtime":
    show_page_loading()
    # Tampilkan halaman Pemeriksaan Realtime
    st.title("Pemeriksaan Realtime")
    st.markdown("---")

    # Tambahkan elemen-elemen yang diperlukan untuk pemeriksaan realtime
    st.write("Dalam mode pemeriksaan realtime, kamera akan digunakan untuk mendeteksi kondisi paru-paru.")




    # Tampilkan prediksi terakhir
    prediction_text = st.empty()

    # Setup WebRTC streamer dengan menggunakan argumen yang diperbarui
    webrtc_ctx = webrtc_streamer(key="example",
                             video_processor_factory=VideoTransformer,
                             rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                             media_stream_constraints={"video": True, "audio": False},
                             async_processing=True)


elif halaman_terpilih == "Edit Gambar":
    show_page_loading()
    # Tampilkan halaman Pemeriksaan Realtime
    st.title("Edit Gambar")
    st.markdown("---")
    st.write("Halaman Edit gambar ini untuk menyoroti dan menganalisis aspek penting dari gambar rontgen Anda, membuka wawasan baru dalam setiap detail.")

    drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
    )
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    if drawing_mode == 'point':
        point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.file_uploader("Background image:", type=["png", "jpg"])

    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # membuat canvas
    if bg_image:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(bg_image) if bg_image else None,
            update_streamlit=realtime_update,
            height=550,
            width=850,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
        )


        



else:
    show_page_loading() 
    st.title("Visualisasi Model AI")
    st.markdown("---")
 

    # Fungsi untuk menampilkan gambar dalam tabel
    def display_image_table(image_path1, title1, caption1, image_path2, title2, caption2):
        image1 = Image.open(image_path1)
        image2 = Image.open(image_path2) if image_path2 else None

        col1, col2 = st.columns(2)
        
        with col1:
            col1.markdown(f'<h2 style="text-align:center;">{title1}</h2>', unsafe_allow_html=True)
            
            # Center-align the image
            col1.markdown(
                f'<div style="display: flex; justify-content: center;"></div>',
                unsafe_allow_html=True
            )
            col1.image(image1, use_column_width=True)
            col1.markdown(f'<p style="text-align:left;">{caption1}</p>', unsafe_allow_html=True)

        with col2:
            if image2:
                col2.markdown(f'<h2 style="text-align:center;">{title2}</h2>', unsafe_allow_html=True)
                
                col2.markdown(
                    f'<div style="display: flex; justify-content: center;"></div>',
                    unsafe_allow_html=True
                )
                col2.image(image2, use_column_width=True)
                col2.markdown(f'<p style="text-align:left;">{caption2}</p>', unsafe_allow_html=True)

    # Daftar gambar, judul, dan caption
    images_info = [
        {'path': 'C:\\Users\\user\\image_TB\\Vocab.png', 'title': 'Vocab Image', 'caption': 'Model AI ini dibagi menjadi data pelatihan (85%) dan data pengujian (15%).'},
        {'path': 'C:\\Users\\user\\image_TB\\Matrix.png', 'title': 'Confusion Matrix', 'caption': 'Confusion Matrix membantu memahami sebuah sistem klasifikasi dengan membandingkan kebenaran prediksi sistem dengan hasil yang diharapkan. Gambar ini merupakan confusion matrix untuk dua kelas kasus: normal dan tuberculosis.'},
        {'path': 'C:\\Users\\user\\image_TB\\Acc.png', 'title': 'Accuracy', 'caption': 'Untuk kelas "tb", akurasi yang dicapai adalah 0.97 dengan 105 sampel.Untuk kelas "non-tb", akurasi yang dicapai adalah 1.0 dengan 525 sampel. Ini menunjukkan bahwa model memiliki kinerja yang sangat baik dalam mengklasifikasikan sampel "non-tb".'},
        {'path': 'C:\\Users\\user\\image_TB\\acc1.png', 'title': 'Accuracy EPOCH', 'caption': 'Model mencapai akurasi hampir 1.0 pada data pelatihan, yang menunjukkan bahwa model dengan baik mempelajari data pelatihan tersebut.Akurasi pada data pengujian berada di sekitar 0.9936, yang cukup tinggi dan menunjukkan bahwa model memiliki generalisasi yang baik pada data yang tidak pernah dilihat sebelumnya.'},
        {'path': 'C:\\Users\\user\\image_TB\\loss.png', 'title': 'Loss EPOCH', 'caption': 'Kehilangan pada data pelatihan menurun hingga mendekati nol, yang menunjukkan bahwa model dengan baik memfitting data pelatihan.Kehilangan pada data pengujian juga menurun, tetapi ada beberapa fluktuasi. Ini mungkin menunjukkan beberapa tanda overfitting, tetapi karena akurasi pengujian masih tinggi, overfitting mungkin tidak terlalu parah.'},
    ]

    # Menampilkan gambar dalam tabel
    for i in range(0, len(images_info), 2):
        if i + 1 < len(images_info):
            display_image_table(
                images_info[i]['path'], images_info[i]['title'], images_info[i]['caption'],
                images_info[i + 1]['path'], images_info[i + 1]['title'], images_info[i + 1]['caption']
            )
        else:
            # Menangani situasi di mana hanya ada satu gambar yang tersisa
            display_image_table(images_info[i]['path'], images_info[i]['title'], images_info[i]['caption'], '', '', '')
