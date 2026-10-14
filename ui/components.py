"""Presentation helpers; render only real project data."""
from html import escape
import math
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from annoloop.reused.training_monitor import ANSI


def hero(kicker, title, description):
    st.markdown(f'<div class="hero"><div class="eyebrow">{escape(kicker)}</div><h2>{escape(title)}</h2><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def empty(title, description):
    st.markdown(f'<div class="empty"><b>{escape(title)}</b><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def chips(names):
    st.markdown(' '.join(f'<span class="chip">{escape(str(n))}</span>' for n in names), unsafe_allow_html=True)


def cards(items):
    for column, (number, title, description) in zip(st.columns(len(items)), items):
        with column:
            st.markdown(f'<div class="card"><div class="eyebrow">{escape(number)}</div><b>{escape(title)}</b><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def box_plot(document, modality):
    """BEV for native 3D boxes; image coordinates for 2D. No invented point cloud."""
    figure = go.Figure()
    for box in document.get('boxes', []):
        values = box['box']
        if modality == '3d':
            x, y, z, length, width, height, yaw = values
            points = np.array([[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2],[-length/2,-width/2]])
            rotation = np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
            points = points @ rotation.T + [x,y]
        else:
            x,y,x2,y2 = values
            points = np.array([[x,y],[x2,y],[x2,y2],[x,y2],[x,y]])
        score = box.get('score')
        label = f"{box['class_name']} · {score:.2f}" if isinstance(score,(int,float)) else box['class_name']
        figure.add_trace(go.Scatter(x=points[:,0],y=points[:,1],mode='lines',name=label,hovertemplate=escape(label)+'<extra></extra>',line=dict(width=2)))
    figure.update_layout(height=420,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor='#ffffff',plot_bgcolor='#f8f9fc',showlegend=False,
                          xaxis_title='X (m)' if modality=='3d' else 'X (pixel)', yaxis_title='Y (m)' if modality=='3d' else 'Y (pixel)')
    figure.update_yaxes(scaleanchor='x',scaleratio=1,autorange=True if modality=='3d' else 'reversed')
    return figure


def tail(path, limit=24000):
    path = Path(path)
    if not path.is_file():
        return 'Chưa có log.'
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-limit))
        return ANSI.sub('', stream.read().decode(errors='replace')).replace('\r', '\n')
