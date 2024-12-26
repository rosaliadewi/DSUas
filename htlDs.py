import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Judul aplikasi
st.title('Model Prediksi Pembatalan Reservasi Hotel')
# Deskripsi aplikasi
st.write("""
    Aplikasi ini membaca data reservasi hotel dari file Excel lokal dan 
    melakukan prediksi apakah reservasi akan dibatalkan atau tidak berdasarkan data yang ada.
""")
# Lokasi file Excel
data= 'Hotel_Reservation_Resampled_Undersampled.xlsx'

try:
    # Membaca dataset
    df = pd.read_excel(data)

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

        # Hapus atau komentari bagian ini untuk tidak menampilkan evaluasi model
        # st.text("Evaluasi Model:")
        # st.text(classification_report(y_test, rf_model.predict(X_test)))

        # Form untuk prediksi data baru
        with st.form("prediction_form"):
            st.subheader("Masukkan Data Baru:")
            no_of_adults = st.number_input("Jumlah Dewasa:", min_value=1, max_value=10)
            no_of_children = st.number_input("Jumlah Anak-anak:", min_value=0, max_value=10)
            no_of_weekend_nights = st.number_input("Malam Akhir Pekan:", min_value=0, max_value=7)
            no_of_week_nights = st.number_input("Malam Mingguan:", min_value=0, max_value=7)
            type_of_meal_plan = st.selectbox("Paket Makanan:", df['type_of_meal_plan'].unique())
            room_type_reserved = st.selectbox("Jenis Kamar:", df['room_type_reserved'].unique())
            lead_time = st.number_input("Lead Time:", min_value=0, max_value=365)
            market_segment_type = st.selectbox("Segmen Pasar:", df['market_segment_type'].unique())
            avg_price_per_room = st.number_input("Harga Rata-rata per Kamar:", min_value=0.0)

            submitted = st.form_submit_button("Prediksi")

            if submitted:
                # Buat DataFrame baru untuk prediksi
                new_data = pd.DataFrame({
                    "no_of_adults": [no_of_adults],
                    "no_of_children": [no_of_children],
                    "no_of_weekend_nights": [no_of_weekend_nights],
                    "no_of_week_nights": [no_of_week_nights],
                    "type_of_meal_plan": [type_of_meal_plan],
                    "room_type_reserved": [room_type_reserved],
                    "lead_time": [lead_time],
                    "market_segment_type": [market_segment_type],
                    "avg_price_per_room": [avg_price_per_room]
                })

                # Encode kategori
                for col in categorical_columns:
                    new_data[col] = label_encoders[col].transform(new_data[col])

                # Scaling data baru
                new_data[numerical_columns] = scaler.transform(new_data[numerical_columns])

                # Prediksi
                prediction = rf_model.predict(new_data)
                result = label_encoder_target.inverse_transform(prediction)[0]

                st.success(f"Hasil Prediksi: {result}")

except FileNotFoundError:   
    st.error(f"File '{file_path}' tidak ditemukan. Pastikan file berada di direktori yang benar.")
