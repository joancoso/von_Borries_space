import streamlit as st
import pandas as pd
import time
import os

def main():

    # Streamlit app title
    st.title("von Borries space of marketed chemicals")

    st.markdown("""
                This apps lets you visualize the space of marketed chemicals.  
                """)

    # Dropdown menu for selecting a molecule
    endpoints = ['Standard', 'PFAS_space']
    reference_space = st.selectbox('Choose reference space',
                                     placeholder='Choose an option',
                                     index=None,
                                     options=endpoints)

    # Upload CSV file
    users_target_chemicals = st.file_uploader("Upload a CSV file with your chemical substances of interest", type="csv")

    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode("utf-8")
    example_csv = pd.read_csv(os.path.join('data','nist_potential_pfas_smiles.tsv'), sep='\t')
    csv = convert_df(example_csv)

    st.sidebar.download_button(
        label="Download example file",
        data=csv,
        file_name="test_space.csv",
        mime="text/csv",
    )

    if users_target_chemicals is not None:
        # Load the uploaded data
        users_target_chemicals = pd.read_csv(users_target_chemicals)

        # Show the input data
        st.write("Uploaded data:", users_target_chemicals)

        from modules.preprocessing import calculate_descriptors_morgan_df
        morgan_df = calculate_descriptors_morgan_df(df=users_target_chemicals, col_smiles="SMILES")
        st.write("Morgan Fingerprints:", morgan_df)

        print('Project substances')

        with st.spinner("Projecting your substances of interest", show_time=True):
            time.sleep(3)

            if reference_space == 'Standard':
                # Show the coordinates data
                coordinates_df = pd.read_csv(os.path.join('data', 'data_market_tsne.csv'))
                # st.write("Coordinates data:", coordinates_df)
                from modules.visualizing import chemical_space_plot_grey
                fig_grey = chemical_space_plot_grey(coordinates_df)
                #fig_color = chemical_space_plot()  @TODO: I will check later because we need to specify hue_column and so on
                st.write(fig_grey)

                # Load the trained object
                # Transform the user to the reference space of interest. 

            elif reference_space == 'PFAS_NIST':
                st.write("We will soon provide the PFAS reference space")
                #@TODO: just load the PFAS coordinates? 
            else:
                st.write("Please choose an option")


        








        st.success("Done!")

        # # Show the predictions
        # st.markdown(""" ### Predictions: """)
        # config = {
        #     "Structure": st.column_config.ImageColumn(width="medium"),
        # }
        # st.dataframe(predictions_df, column_config=config, row_height=100)


        # This is how I could plot the chemical space with plotly for the Shiny App - assumes input where the user selects a variable for hue (color) - Kerstin
        # @render_widget
        # def chemical_space_plot():
        #     df = loaded_data()  # Use cached data
        #     if df.empty:
        #         return px.scatter(title="No data available")

        #     # Define custom order (equivalent to seaborn's hue_order)
        #     hue_column = input.hue_column()

        #     # Set palette
        #     if hue_column in ['Toxicity [log10 mg/kg-d]', 'Uncertainty [95% CI width]']:
        #         hue_order = sorted(df[hue_column].unique())
        #         if hue_column == 'Toxicity [log10 mg/kg-d]':
        #             hue_order = hue_order[::-1]
        #         colorscale = sns.diverging_palette(250, 10, s=100, l=50, sep=1, n=len(hue_order), center="light")
        #         colors = [mcolors.to_hex(c) for c in colorscale]
        #         color_map = dict(zip(hue_order, colors)) 

        #     else:
        #         hue_order = df[hue_column].value_counts().index.tolist() # sort by frequency
        #         df[hue_column] = pd.Categorical(df[hue_column], categories=hue_order, ordered=True)
        
        #         colors = [mcolors.rgb2hex(c) for c in plt.cm.tab20.colors]
        #         # - repeat colors if more than 20 categories
        #         repeat_colors = (colors * (len(hue_order) // len(colors) + 1))[:len(hue_order)]
        #         color_map = dict(zip(hue_order, repeat_colors))

            
        #     # Scatterplot
        #     df.fillna('not assigned', inplace=True)
        #     fig = px.scatter(df, x="TSNE1", y="TSNE2", color=hue_column,
        #                     color_discrete_map=color_map,
        #                     category_orders = {hue_column: hue_order},
        #                     hover_name = 'Chemical name',
        #                     hover_data=['CAS RN', 'Predicted POD [95% CI]', 'Superclass', 'Class', 'Subclass'],
        #                     render_mode="webgl",
        #                     height=700, width=1200
        #                     )
        #     fig.update_traces(showlegend=False)

        #     fig.update_traces(marker=dict(size=3, opacity=0.3, line=dict(width=0))) # plot markers
        #     fig.update_traces(marker=dict(size=10), opacity=1, selector=dict(legendgroup=True)) # legend markers

        #     # Add custom legend traces to control the opacity of the legend markers
        #     categories = hue_order

        #     for category in categories:
        #         fig.add_trace(go.Scatter(
        #             x=[None], y=[None],  # Invisible points
        #             mode='markers',
        #             marker=dict(color=color_map[category], size=12, opacity=1),  # Full opacity for the legend
        #             legendgroup=category,
        #             showlegend=True,
        #             name=category,
        #         ))

        #     fig.update_layout(
        #         dragmode="zoom",  # - enables zooming
        #         uirevision=True,  # Track the zoom and pan state
        #         xaxis=dict(title="TSNE1",visible=False, showgrid=False, zeroline=False, 
        #                 fixedrange=False),
        #         yaxis=dict(title="TSNE2",visible=False, showgrid=False, zeroline=False, 
        #                 fixedrange=False),
        #         modebar=dict(add=["zoom", "pan", "resetScale"]),
        #         plot_bgcolor="rgba(0,0,0,0)",
        #         paper_bgcolor="rgba(0,0,0,0)",
        #         legend_title_text=hue_column,
        #         legend=dict(itemsizing='constant',  # Control legend item sizing
        #                     itemwidth=30,  # Adjust this value to control the item width
        #                     tracegroupgap=0,  # Gap between legend items
        #                     x=1.0,  # Position the legend towards the right
        #                     y=0.5,  # Center vertically
        #                     xanchor='left',
        #                     yanchor='middle',
        #                     orientation='v',  # Arrange legend vertically
        #                     font=dict(family="Arial", size=10)))
                        
        #     return fig



if __name__ == '__main__':
    main()
    print('app is running')