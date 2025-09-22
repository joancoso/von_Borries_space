import streamlit as st
import pandas as pd
import time

def main():

    # Streamlit app title
    st.title("von Borries space of marketed chemicals")

    st.markdown("""
This apps lets you visualize the space of marketed chemicals.  
    """)

    # Dropdown menu for selecting a molecule
    endpoints = ['Standard', 'PFAS_space']
    model_selected_from_box = st.selectbox('Choose endpoint to predict',
                                     placeholder='Choose an option',
                                     index=None,
                                     options=endpoints)

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a CSV file with chemical substance data", type="csv")

    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")
    example_csv = pd.read_csv('test_market_chemicals_space.csv')
    csv = convert_df(example_csv)

    st.sidebar.download_button(
        label="Download example file",
        data=csv,
        file_name="test_space.csv",
        mime="text/csv",
    )

    if uploaded_file is not None:
        # Load the uploaded data
        input_data = pd.read_csv(uploaded_file)

        # Show the input data
        st.write("Uploaded data:", input_data)

        print('Start predictions')

        with st.spinner("Prediction is running...", show_time=True):
            time.sleep(3)

            if model_selected_from_box == 'Standard':
                from src.modeling import transform_target
                space_coordinates_df = transform_target(input_data)
            elif model_selected_from_box == 'PFAS_space':
                pass  # Placeholder for future implementation
            else:
                st.write("Please choose an option")

        st.success("Done!")

        # # Show the predictions
        # st.markdown(""" ### Predictions: """)
        # config = {
        #     "Structure": st.column_config.ImageColumn(width="medium"),
        # }
        # st.dataframe(predictions_df, column_config=config, row_height=100)


if __name__ == '__main__':
    main()
    print('app is running')