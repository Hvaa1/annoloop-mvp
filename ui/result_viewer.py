"""Read-only saved final/ensemble viewer, including rejected boxes and real XYZ."""
from collections import Counter
from html import escape
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from annoloop.io import read,inside
from components import box_plot
from prediction_page import cloud

REASONS={'unmatched':'Chưa ghép được giữa model','quality_conflict':'Một model tin, model kia không',
         'low_calibrated_quality':'Dưới ngưỡng confidence/quality','insufficient_points':'Thiếu point hỗ trợ',
         'insufficient_calibrated_evidence':'Chưa đủ bằng chứng đã calibration',
         'cross_class_conflict':'Xung đột class'}


def filtered(document,classes,mode):
    """Display policy is separate from export/training policy; never mutate boxes."""
    accepted=[b for b in document.get('boxes',[]) if b['class_name'] in classes]
    ignored=[b for b in document.get('ignored',[]) if b['class_name'] in classes]
    if mode=='Chỉ box được giữ':ignored=[]
    elif mode=='Chỉ ignore':accepted=[]
    return accepted,ignored


def figure_for(accepted,ignored,points=None):
    boxes=accepted+ignored
    figure=box_plot(dict(boxes=boxes),'3d')
    for index,(trace,box) in enumerate(zip(figure.data,boxes)):
        is_ignore=index>=len(accepted)
        status='ignore' if is_ignore else 'được giữ'
        trace.update(line=dict(color='#cf851b' if is_ignore else '#7054d8',
                               dash='dash' if is_ignore else 'solid',width=1.5 if is_ignore else 2.5))
        trace.name=box['class_name']+' · '+status
        details=[trace.name]
        for field in ('confidence','quality','score','point_count'):
            if field in box:details.append(field+': '+str(box[field]))
        if box.get('reason'):details.append('Lý do: '+REASONS.get(box['reason'],box['reason']))
        trace.hovertemplate='<br>'.join(escape(s) for s in details)+'<extra></extra>'
    if points is not None:
        scatter=go.Scattergl(x=points[:,0],y=points[:,1],mode='markers',name='Point cloud',
            marker=dict(size=1,color='#969eac',opacity=.4),hoverinfo='skip')
        figure=go.Figure(data=[scatter,*figure.data],layout=figure.layout)
        if len(points):
            # Focus on the dense cloud plus every box; sparse far returns only shrink the view.
            low,high=np.percentile(points[:,:2],[5,95],axis=0)
            for box in boxes:
                centre=np.asarray(box['box'][:2],float);low=np.minimum(low,centre);high=np.maximum(high,centre)
            middle,half=(low+high)/2,float(max(high-low))/2+5
            figure.update_xaxes(range=[float(middle[0]-half),float(middle[0]+half)],autorange=False)
            figure.update_yaxes(range=[float(middle[1]-half),float(middle[1]+half)],autorange=False,scaleanchor='x',scaleratio=1)
            figure.update_layout(height=600)
    figure.update_layout(uirevision='saved_result_bev')
    return figure


@st.cache_data(show_spinner=False,max_entries=12)
def counts(root,frames,stamps):
    accepted=Counter();ignored=Counter();reasons=Counter()
    for row in frames:
        document=read(inside(root,row['file']))
        if document.get('frame_id')!=row['frame_id']:
            raise ValueError('Frame kết quả không khớp chỉ mục đã lưu')
        accepted.update(b['class_name'] for b in document.get('boxes',[]))
        ignored.update(b['class_name'] for b in document.get('ignored',[]))
        reasons.update((b['class_name'],b.get('reason','unknown')) for b in document.get('ignored',[]))
    return dict(accepted),dict(ignored),[(name,reason,count) for (name,reason),count in reasons.items()]


def render(store,project,root,result,namespace,classes=None):
    root=Path(root);pid=project['id'];key=namespace+'_'+pid+'_'+root.name
    frames=result['frames']
    if not frames:
        st.info('Kết quả này chưa có frame đã lưu.');return
    spec_path=root/'spec.json';spec=read(spec_path) if spec_path.exists() else {}
    if spec and spec.get('project_id')!=pid:
        raise ValueError('Kết quả thuộc dự án khác; không ghép vào dự án đang xem')
    inventory=read(store.path(pid)/'dataset.json');media={f['frame_id']:f for f in inventory['frames']}
    if any(row['frame_id'] not in media for row in frames):
        raise ValueError('Frame kết quả nằm ngoài dataset dự án')
    stamps=[]
    for row in frames:
        stat=inside(root,row['file']).stat();stamps.append((stat.st_mtime_ns,stat.st_size))
    accepted,ignored,reasons=counts(str(root),frames,stamps)
    names=list(dict.fromkeys([*(classes or result.get('class_names',project['schema']['classes'])),*accepted,*ignored]))
    st.caption('Dự án: '+project['name']+' · '+pid[:8]+' · phiên bản: '+result.get('version','ensemble')+' · lượt: '+root.name[:8])
    st.dataframe([{'Class':name,'Box được giữ':accepted.get(name,0),'Box ignore':ignored.get(name,0)} for name in names],
                 hide_index=True,width='stretch')
    unique=len({media[f['frame_id']]['sha256'] for f in frames})
    st.caption(f'{len(frames)} mục frame / {unique} nội dung file khác nhau theo SHA256. Số box là số dự đoán, chưa phải số vật thể độc lập hay độ chính xác.')
    selected=st.multiselect('Class kết quả hiển thị',names,default=names,key='result_classes_'+key)
    mode=st.radio('Box kết quả hiển thị',['Được giữ + ignore','Chỉ box được giữ','Chỉ ignore'],
                  horizontal=True,key='result_policy_'+key)
    fid=st.selectbox('Frame kết quả đã lưu',[f['frame_id'] for f in frames],key='result_frame_'+key)
    row=next(f for f in frames if f['frame_id']==fid);document=read(inside(root,row['file']))
    keep,review=filtered(document,selected,mode)
    a,b=st.columns(2);a.metric('Box được giữ trên frame',len(keep));b.metric('Box ignore trên frame',len(review))
    points=None;frame=media[fid]
    try:
        columns=spec.get('config',{}).get('bin_columns')
        stat=Path(frame['media']).stat()
        points,total=cloud(frame['media'],frame['sha256'],columns,stat.st_mtime_ns,stat.st_size)
        st.caption(f'Point cloud thật: {total:,} điểm, hiển thị {len(points):,} điểm mẫu · BEV nhìn từ trên xuống.')
    except (OSError,ValueError,ImportError) as error:
        st.warning('Chưa đọc được point cloud: '+str(error)+'. Hình học box vẫn hiển thị.')
    st.plotly_chart(figure_for(keep,review,points),width='stretch',key='result_plot_'+key)
    st.caption('Tím nét liền: được giữ · Cam nét đứt: ignore. Bộ lọc chỉ thay đổi phần xem, giữ nguyên nhãn train và gói CVAT.')
    if not keep and not review:st.info('Không có box khớp bộ lọc trên frame này. Có thể đổi class hoặc hiển thị ignore.')
    with st.expander('Confidence / quality và lý do ignore'):
        rows=[]
        for state,boxes in [('Được giữ',keep),('Ignore',review)]:
            for box in boxes:
                rows.append({'Class':box['class_name'],'Trạng thái':state,'Confidence':box.get('confidence'),
                             'Quality':box.get('quality'),'Score gốc':box.get('score'),
                             'Point':box.get('point_count'),'Lý do':REASONS.get(box.get('reason'),box.get('reason',''))})
        st.dataframe(rows,hide_index=True,width='stretch')
        st.json(dict(frame_id=fid,boxes=keep,ignored=review))
    with st.expander('Thống kê lý do ignore trên toàn bộ kết quả'):
        st.dataframe([{'Class':name,'Lý do':REASONS.get(reason,reason),'Số dự đoán':count}
                      for name,reason,count in reasons if name in selected],hide_index=True,width='stretch')


def final_panel(store,project):
    from annoloop import final_delivery,local_ensemble
    from annoloop.io import digest
    complete=[p.parent for p in final_delivery.jobs(store,project['id'])
              if local_ensemble.status(p.parent)['status']=='completed' and (p.parent/'result.json').exists()]
    if not complete:return
    with st.expander('Kết quả final đã lưu · xem box được giữ và ignore',expanded=True):
        root=st.selectbox('Lượt final để xem',complete,format_func=lambda p:read(p/'result.json')['version']+' · '+p.name[:8])
        state=read(root/'state.json')
        if digest(root/'result.json')!=state.get('result_sha256'):
            raise ValueError('Kết quả final đã thay đổi')
        render(store,project,root,read(root/'result.json'),'compare_final')
