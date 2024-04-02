import streamlit as st
from inference import Lyrics_Generator
import tensorflow as tf

# container = st.container(border=True)
st.set_page_config(page_title="Lyrics Generator", page_icon="./assets/person.png")
# st.sidebar.write("Title")
# Load Model
model_load = tf.keras.models.load_model("./model/Lyrics_Generator_v2.h5")

title = st.title("Kel 5 - Lyrics Generator Using BERT")

st.image("./assets/person.png", width=400)

st.write('Label lirik diambil dalam segmen 40 kata, menjadi dasar untuk menciptakan kata-kata berikutnya. Pendekatan ini memastikan keluaran artistik dan matematis.') 

# Create a text input 
st.subheader('Masukkan Teks (40 Kata):')
user_input = st.text_input('Masukkan Lirik awal...', value="bersama ku mengerti kau dengan dia hari ") 

generate_button = False
ftext = ""
check_size = len(user_input)
if len(user_input) == 40:
    generate_button = st.button("Generate Lyrics")
else:
    ftext = 'Masukkan harus terdiri dari tepat 40 karakter. Jumlah Karakter anda : {}'.format(check_size)
    st.warning(ftext)

if generate_button:
    if user_input:
        song_1 = Lyrics_Generator(user_input, 400, model_load)
        st.subheader("Generated Lyrics:")
        print(song_1)
        # Memisahkan teks menjadi potongan-potongan 50 karakter
        split_lyrics = [song_1[i:i+40] for i in range(0, len(song_1), 40)]
        # Menggabungkan kembali potongan-potongan dengan spasi di antara mereka
        formatted_lyrics = "".join(split_lyrics)
        
        st.text(formatted_lyrics)
    else:
        st.warning("Please enter a starting phrase.")
        
