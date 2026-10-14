"""Real checkpoint validation/sample prediction without manual file entry."""
from pathlib import Path
import streamlit as st
from annoloop.inspection import checkpoint_choices,start_inspection,inspection_status
import numpy as np
from annoloop.io import read
from components import hero,empty,box_plot,tail
import plotly.graph_objects as go


def model_inspection(store,settings,p):
    hero('KIỂM TRA MODEL','Dùng checkpoint đã lưu, xem kết quả thật.', 'Đánh giá trên source validation có nhãn hoặc xem dự đoán target. Các thao tác này chỉ inference, không train lại.')
    candidates=checkpoint_choices(store,p['id'],settings)
    if not candidates:
        empty('Chưa có checkpoint của dự án','Sau khi có checkpoint trên đĩa, chọn tại đây để đánh giá hoặc predict.');return
    lookup={c['path']:c for c in candidates}
    selected=st.selectbox('Checkpoint cần kiểm tra',list(lookup),format_func=lambda path:lookup[path]['name'])
    choice=lookup[selected]
    if choice['intermediate']:st.info('Checkpoint trung gian của run đã dừng/chưa hoàn tất. Có thể kiểm tra; chưa tự đưa vào model chính của dự án.')
    st.caption('Class trong checkpoint: '+', '.join(choice['classes']))
    a,b=st.columns(2)
    score=a.slider('Score tối thiểu',0.,1.,.3,step=.05)
    iou=b.slider('IoU 3D để ghép source GT',.1,.9,.5,step=.05)
    st.caption('Precision/recall tính theo ngưỡng này; đây không phải mAP chính thức nuScenes. Score hiện là score model, chưa calibration.')
    confirmed=st.checkbox('Target dùng mét; X hướng trước, Y trái, Z lên, cùng hệ tọa độ LiDAR đã chuẩn hóa')
    bin_columns=st.selectbox('Nếu target là BIN: số cột mỗi điểm',[4,3,5],format_func=lambda x:{3:'3 · XYZ',4:'4 · XYZI',5:'5 · XYZIT'}[x])
    parent=store.path(p['id'])/'inspections'
    active=any(inspection_status(path.parent)['process_alive'] for path in parent.glob('*/inspection.json'))
    a,b=st.columns(2)
    if a.button('Đánh giá source validation',type='primary',disabled=active):
        with st.spinner('Kiểm tra runtime và gửi job đánh giá…'):start_inspection(store,p['id'],selected,'validation',settings,score,iou)
        st.rerun()
    if b.button('Predict sample target',disabled=active or not confirmed):
        with st.spinner('Kiểm tra runtime và gửi job predict…'):start_inspection(store,p['id'],selected,'target_sample',settings,score,iou,confirmed,bin_columns)
        st.rerun()
    st.caption('Sample dùng các frame cố định của dự án. Không đọc annotation/gold target. GPU chạy một job tại một thời điểm.')
    results_fragment(parent)


@st.fragment(run_every=5)
def results_fragment(parent):
    paths=sorted(parent.glob('*/inspection.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not paths:return
    selected=st.selectbox('Lượt kiểm tra',[str(p.parent) for p in paths],format_func=lambda p:('Validation' if read(Path(p)/'inspection.json')['mode']=='validation' else 'Target sample')+' · '+Path(p).name[:8])
    root=Path(selected);state=inspection_status(root)
    st.write('Trạng thái: **'+state['status']+'**')
    if state.get('total_frames'):st.progress(min(1.,state.get('completed_frames',0)/state['total_frames']),text=f"{state.get('completed_frames',0)}/{state['total_frames']} frame")
    if state.get('error'):st.error(state['error'])
    with st.expander('Log kiểm tra',expanded=state['status'] in ('running','failed')):st.code(tail(root/'inspection.log'),language='log',height=180)
    if state['status']!='completed':return
    result=read(root/'result.json')
    if result['mode']=='validation':
        st.subheader('Kết quả source validation')
        st.dataframe(result['metrics'],width='stretch',hide_index=True)
        st.caption(f"3D IoU ≥ {result['iou_threshold']:.2f} · Score ≥ {result['score_threshold']:.2f} · {len(result['frames'])} frame · Source GT, không phải target gold")
    else:
        items={f['frame_id']:f for f in result['frames']}
        frame=st.selectbox('Frame dự đoán',list(items))
        item=items[frame];document=read(root/item['file'])
        figure=box_plot(document,'3d')
        sampled=np.load(root/item['point_preview'],allow_pickle=False)
        figure.add_trace(go.Scattergl(x=sampled[:,0],y=sampled[:,1],mode='markers',marker=dict(size=1,color='#9b9fab'),name='LiDAR',hoverinfo='skip'))
        st.plotly_chart(figure,width='stretch')
        st.caption(f"{item['point_count']:,} điểm · {len(document['boxes'])} box · BEV trong hệ tọa độ target")
        with st.expander('Class, score và box'):st.json(document)
    st.download_button('Tải kết quả JSON',(root/'result.json').read_bytes(),file_name='inspection_result.json',mime='application/json')
