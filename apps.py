import streamlit as st
import pandas as pd
import pickle
import datetime


st.title("🎓 Jaya Jaya Institute Student Prediction using Machine Learning")

tab1, tab2, tab3 = st.tabs([
    "Student Assessment",
    "Dashboard",
    "About"
])

with tab1:
    st.header("Student Prediction")

    # Load model
    def load_model(model_name):
        if model_name == 'Random Forest':
            model = pickle.load(open('models/Random Forest.pkl', 'rb'))
        elif model_name == 'Decision Tree':
            model = pickle.load(open('models/Decision Tree.pkl', 'rb'))
        elif model_name == 'Logistic Regression':
            model = pickle.load(open('models/Logistic Regression.pkl', 'rb'))
        elif model_name == 'SVM':
            model = pickle.load(open('models/SVM.pkl', 'rb'))
        elif model_name == 'XGB':
            model = pickle.load(open('models/XGB.pkl', 'rb'))
        elif model_name == 'GBM':
            model = pickle.load(open('models/Gradient Boosting.pkl', 'rb'))       
        return model


    # Fungsi untuk melakukan prediksi
    def predict_status(model, data):
        predict = model.predict(data)
        return predict

    # Fungsi untuk mewarnai prediksi
    def cpred(col):
        color = 'red' if col == 'Dropout' else 'green'
        return f'color: {color}'


    
        
    with st.expander("### 📌 How to run the prediction:"):
        st.markdown(
            """
            - 🏷 Choose machine learning model
            - 📂 Upload `Filetest.csv`
            - 🚀 Click predict button  
            - 📥 Result will appear and can be 'Download (.csv)'
            """, unsafe_allow_html=True
        )



    
    def main():
        #st.title('Jaya Jaya Institute Student Prediction using Machine Learning')
        





        #with st.expander("How to run the prediction:"):
        #    st.write(
        #        """
        #            1. Choose machine learning model
        #            2. Upload Filetest.csv
        #            3. Click predict button
        #            4. Result will appear and can be 'Download (.csv)'. 
        #        """
        #    )



        # Pemilihan model ML
        #model_name = st.radio("Choose Machine Learning Model", ("Random Forest", "Decission Tree", "Logistic Regression", "XGB", "GBM", "SVM"
        #         #                                                           ))



        
        st.markdown("### Choose a Machine Learning Model 🤖 ")
        model_name = st.radio("", 
                        ("Random Forest", "Decision Tree", "Logistic Regression", 
                        "XGB", "GBM", "SVM"))



        # Upload File
        #upload = st.file_uploader("Upload Filetest", type=["csv"])


        st.markdown("📁 Upload Filetest:")
        upload = st.file_uploader(" ", type=["csv"])

        if upload is not None:
            data = pd.read_csv(upload)

            st.write("Data test:")
            counting = st.slider("Choose number of student", 1, len(data), 5)
            st.write(data.head(counting))

            # ID dan StudentName
            ID = data['ID']
            StudentName = data['Name']
            data = data.drop(columns=['ID', 'Name'])

            # Load model
            model = load_model(model_name)

            # click button
            if st.button('✨Predict'):
                # Prediction
                predict = predict_status(model, data)

                # Labelling
                labelling = ['Graduate' if pred == 1 else 'Dropout' for pred in predict]

                # Result
                result = pd.DataFrame({
                    'ID': ID,
                    'Name': StudentName,
                    'Status Prediction': labelling
                })

                
                st.write("Prediction result:") 
                st.dataframe(result.style.map(cpred, subset=['Status Prediction']))

                
                csv = result.to_csv(index=False) # Download result
                st.download_button(
                    label="Download Prediction Result 📥",
                    data=csv,
                    file_name='Prediction result.csv',
                    mime='text/csv'
                )

    if __name__ == '__main__':
        main()




with tab2:
    st.header("tab2")







year_now = datetime.date.today().year
year = year_now if year_now == 2025 else f'2025 - {year_now}'
name = "[B244044F]"
copyright = 'Copyright © ' + str(year) + ' ' + name
st.caption(copyright)