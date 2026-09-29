"""Optional standalone Plotly scatterplot for the correlation report."""


def plot(result, path):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    colors = [color for color in ('white','black') if color in result['results']]
    titles = []
    for color in colors:
        statistics = result['results'][color]
        limits = statistics['confidence_intervals_95']['pearson']['bounds']
        interval = f'[{limits[0]:.2f}, {limits[1]:.2f}]' if limits else 'unavailable'
        titles.append(f"{color.title()}: correlation {statistics['pearson']:.2f}<br>Approx. 95% CI {interval}; {statistics['cluster_count']} groups")
    fig = make_subplots(rows=1,cols=len(colors),shared_yaxes=True,subplot_titles=titles,horizontal_spacing=.08)
    for column,color in enumerate(colors,1):
        tint = '#27669b' if color=='white' else '#985826'
        sample = [c for c in result['chapters'] if c['color']==color]
        xs,ys = [c['depth'] for c in sample],[c['delta'] for c in sample]
        fig.add_trace(go.Scatter(x=xs,y=ys,mode='markers',name=color.title(),showlegend=False,
            marker=dict(size=10,color=tint,opacity=.85,line=dict(width=1,color='white')),
            customdata=[[c['name'],100*c['reach'],100*c['score'],100*c['baseline'],c['cluster']] for c in sample],
            hovertemplate='%{customdata[0]}<br>Depth: %{x:.2f} own moves<br>Delta: %{y:.2f} pp<br>Chapter reach: %{customdata[1]:.2f}%<br>Score: %{customdata[2]:.2f}%<br>Entry baseline: %{customdata[3]:.2f}%<br>Resampling group: %{customdata[4]}<extra></extra>'),row=1,col=column)
        slope = result['results'][color]['slope_pp_per_move']
        if slope is not None:
            intercept = sum(ys)/len(ys)-slope*sum(xs)/len(xs)
            xx = [min(xs),max(xs)]
            fig.add_trace(go.Scatter(x=xx,y=[slope*x+intercept for x in xx],mode='lines',showlegend=False,
                                    line=dict(color=tint,width=2),hoverinfo='skip'),row=1,col=column)
        fig.update_xaxes(title='Expected prepared depth (own moves after entry)',row=1,col=column)
        fig.update_yaxes(gridcolor='#e5e9ee',zerolinecolor='#919da8',row=1,col=column)
    fig.update_yaxes(title='Repertoire score minus entry baseline (pp)',row=1,col=1)
    fig.update_layout(template='plotly_white',height=700,title='Prepared depth and modeled score improvement',
                      margin=dict(l=70,r=35,t=125,b=125),font=dict(family='Arial',size=13))
    fig.add_annotation(text='Intervals resample shared-theory groups, conditional on the saved chapter estimates.<br>Few groups and remaining dependence limit reliability. Lichess count uncertainty is not included.<br>Hover for exact values. Lines are descriptive fits, with no confidence bands.',
                       x=0,y=-.24,xref='paper',yref='paper',showarrow=False,xanchor='left',align='left',font=dict(size=12,color='#52616e'))
    fig.write_html(path,include_plotlyjs=True,full_html=True,config=dict(displaylogo=False,responsive=True))
