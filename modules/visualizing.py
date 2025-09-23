import plotly.express as px
import os
import pandas as pd

def get_color_class_mapping():
    df_market_tsne = pd.read_csv(os.path.join('data', 'data_market_tsne.csv'))
    # This is the palette I used in my publication for the top 15 most common ClassyFire Superclasses + some code snippets
    palette = ['darkslategrey', 'teal', 'aquamarine', 'darkred', 'orangered', 'mediumpurple', 'darkorchid',
                'mediumblue', 'royalblue', 'skyblue', 'darkgoldenrod', 'darkorange', 'gold',  'lightpink', 'hotpink',
                'lightgrey']

    top15 = df_market_tsne.groupby('Superclass').count()['TSNE1'].sort_values(ascending=False).index[:15]
    df_market_tsne['Superclass (top 15)'] = df_market_tsne['Superclass'].where(df_market_tsne['Superclass'].isin(top15), 'Other')
    hue_order = top15.sort_values().to_list() + ['Other']

    return palette, top15, hue_order


def chemical_space_plot(df, hue_column, color_map, hover_name = 'PREFERRED_NAME', hover_data=['CASRN', 'Superclass', 'Class', 'Subclass'], train=False):
    # Scatterplot
    df.fillna('not assigned', inplace=True)
    fig = px.scatter(df, x="TSNE1", y="TSNE2", color=hue_column,
                    color_discrete_map=color_map,
                    hover_name = hover_name,
                    hover_data=hover_data,
                    render_mode="webgl",
                    height=700, width=1200
                    )
    fig.update_traces(showlegend=False)

    if train==True:
        symbol='diamond',
        fig.update_traces(marker=dict(size=5, symbol='diamond' ,opacity=0.5, line=dict(width=0)))
    else:
        symbol='circle',
        fig.update_traces(marker=dict(size=3, opacity=0.3, line=dict(width=0))) # plot markers

    # Add custom legend traces to control the opacity of the legend markers
    for c in color_map:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(color=color_map[c], symbol=symbol, size=12, opacity=1),
            legendgroup=c,
            showlegend=True,
            name=c,
        ))

    fig.update_layout(
        dragmode="zoom",
        uirevision=True,
        xaxis=dict(title="TSNE1",visible=False, showgrid=False, zeroline=False, 
                fixedrange=False),
        yaxis=dict(title="TSNE2",visible=False, showgrid=False, zeroline=False, 
                fixedrange=False),
        modebar=dict(add=["zoom", "pan", "resetScale"]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text=hue_column,
        legend=dict(itemsizing='constant',
                    itemwidth=30,  
                    tracegroupgap=0,
                    x=1.0,
                    y=0.5,
                    xanchor='left',
                    yanchor='middle',
                    orientation='v',
                    font=dict(family="Arial", size=10)))
                
    return fig


# might be useful to keep overall space in background when embedding a new data set
def chemical_space_plot_grey(df):
    palette, top15, hue_order = get_color_class_mapping()
    
    fig_grey = px.scatter(df, x="TSNE1", y="TSNE2", hover_name = 'PREFERRED_NAME', hover_data=['CASRN', 'Superclass', 'Class', 'Subclass'],
                    render_mode="webgl", height=700, width=1200)

    fig_grey.update_traces(
        marker=dict(color='lightgrey',size=3, opacity=0.3, line=dict(width=0)),
        name='grey_market')

    fig_grey.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font_color='black',
        xaxis=dict(title="TSNE1",visible=False, showgrid=False, zeroline=False, 
                    fixedrange=False),
        yaxis=dict(title="TSNE2",visible=False, showgrid=False, zeroline=False, 
                    fixedrange=False))

    return fig_grey
