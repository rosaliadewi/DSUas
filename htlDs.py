import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Judul aplikasi
st.title('Model Prediksi Pembatalan Reservasi Hotel')

# Lokasi file Excel
file_path = 'C:/Users/INFINIX/Documents/Semester 5/HotelDs/Hotel_Reservation_Resampled_Undersampled.xlsx'

try:
    # Membaca dataset
    df = pd.read_excel(file_path)

    # Periksa kolom target
    if 'booking_status' not in df.columns:
        st.error("Kolom 'booking_status' tidak ditemukan dalam dataset.")
    else:
        # Pisahkan fitur dan target
        X = df.drop(columns=['booking_status', 'Booking_ID', 'arrival_date'], errors='ignore')
        y = df['booking_status']

        # Encode target
        label_encoder_target = LabelEncoder()
        y = label_encoder_target.fit_transform(y)

        # Encode fitur kategori
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        for col in categorical_columns:
            label_encoders[col] = LabelEncoder()
            X[col] = label_encoders[col].fit_transform(X[col])

        # Scaling data numerik
        numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
        scaler = StandardScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

        # Split data untuk pelatihan dan pengujian
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)

        # Evaluasi model (dihapus atau dikomentari)
        # st.text("Evaluasi Model:")
        # st.text(classification_report(y_test, rf_model.predict(X_test)))

        # Inisialisasi tabel data baru (dengan hasil prediksi)
        if "new_data_with_inputs" not in st.session_state:
            st.session_state.new_data_with_inputs = pd.DataFrame()

        # Form untuk input data baru
        with st.form("input_form", clear_on_submit=True):  # Menambahkan clear_on_submit=True
            st.subheader("Masukkan Data Baru:")
            no_of_adults = st.number_input("Jumlah Dewasa:", min_value=1, max_value=10, key="adults")
            no_of_children = st.number_input("Jumlah Anak-anak:", min_value=0, max_value=10, key="children")
            no_of_weekend_nights = st.number_input("Malam Akhir Pekan:", min_value=0, max_value=7, key="weekend_nights")
            no_of_week_nights = st.number_input("Malam Mingguan:", min_value=0, max_value=7, key="week_nights")
            type_of_meal_plan = st.selectbox("Paket Makanan:", df['type_of_meal_plan'].unique(), key="meal_plan")
            room_type_reserved = st.selectbox("Jenis Kamar:", df['room_type_reserved'].unique(), key="room_type")
            lead_time = st.number_input("Lead Time:", min_value=0, max_value=365, key="lead_time")
            market_segment_type = st.selectbox("Segmen Pasar:", df['market_segment_type'].unique(), key="market_segment")
            avg_price_per_room = st.number_input("Harga Rata-rata per Kamar:", min_value=0.0, key="avg_price")

            submitted = st.form_submit_button("Tambahkan dan Prediksi")

            if submitted:
                # Simpan data input sesuai format asli
                original_data = {
                    "no_of_adults": no_of_adults,
                    "no_of_children": no_of_children,
                    "no_of_weekend_nights": no_of_weekend_nights,
                    "no_of_week_nights": no_of_week_nights,
                    "type_of_meal_plan": type_of_meal_plan,
                    "room_type_reserved": room_type_reserved,
                    "lead_time": lead_time,
                    "market_segment_type": market_segment_type,
                    "avg_price_per_room": avg_price_per_room
                }

                # Buat DataFrame dari inputan pengguna
                new_data = pd.DataFrame([original_data])

                # Data untuk prediksi (diolah)
                pred_data = new_data.copy()
                for col in categorical_columns:
                    pred_data[col] = label_encoders[col].transform(pred_data[col])
                pred_data[numerical_columns] = scaler.transform(pred_data[numerical_columns])

                # Prediksi
                prediction = rf_model.predict(pred_data)
                result = label_encoder_target.inverse_transform(prediction)[0]

                # Tambahkan hasil prediksi ke data input asli
                original_data['booking_status'] = result
                new_data_with_prediction = pd.DataFrame([original_data])

                # Simpan ke tabel session_state
                st.session_state.new_data_with_inputs = pd.concat(
                    [st.session_state.new_data_with_inputs, new_data_with_prediction],
                    ignore_index=True
                )

                st.success("Data berhasil ditambahkan dan diprediksi!")

        # Tampilkan tabel dengan data input asli dan hasil prediksi
        st.subheader("Tabel Data Baru dengan Hasil Prediksi:")
        st.write(st.session_state.new_data_with_inputs)

except FileNotFoundError:
    st.error(f"File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang benar.")
