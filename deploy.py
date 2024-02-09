import streamlit as st
import joblib

from hypermodel import hypermodel

def main():
    st.title('BREAST CANCER DETECTION')

    test_result = st.text_input('Enter the Test Results')

    if st.button('Predict'):
        if not test_result:
            st.warning('Please Enter the Test Results')
        else:
            test_result_list = [float(val.strip()) for val in test_result.split(',')]
            data_is_legit = hypermodel(test = test_result_list)
            is_legit = data_is_legit.is_legit

            if is_legit == True:
                st.success(f'BENIGN CANCER')
            else:
                st.error(f'MALIGNANT CANCER')

if __name__ == '__main__':
    main()
