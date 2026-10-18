"""Saved prediction viewer. CPU point-cloud display only; no GPU dispatch."""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from annoloop import previews
from annoloop.io import read
from components import box_plot, live_fragment


@st.cache_data(show_spinner=False, max_entries=16)
def cloud(media, checksum, columns, modified, size):
    return previews.point_preview({'media': media, 'sha256': checksum}, columns)


def saved_predictions(store, project):
    from annoloop.adaptation import active_job
    _saved_predictions(store, project, live=active_job(store, project['id']) is not None)


@live_fragment(5)
def _saved_predictions(store, project):
    latest = store.get(project['id'])
    if any(latest[k] != project[k] for k in ('completed_rounds', 'selected_version')):
        st.rerun(scope='app')
    attempts = previews.saved_attempts(store, project['id'])
    if not attempts:
        return
    st.subheader('Dự đoán A/B trước và sau train')
    choices = {a['id']: a for a in attempts}
    preferred = next((i for i, a in enumerate(attempts)
                      if any(r['available'] for r in a['phases']['after'].values())), 0)
    selected = st.selectbox('Lượt dự đoán đã lưu', list(choices),
        format_func=lambda key: choices[key]['version'] + ' · ' + Path(choices[key]['job']).name[:8]
                    + ' · ' + choices[key]['state']['status'], index=preferred, key='preview_attempt_' + project['id'])
    attempt = choices[selected]
    if attempt['trained']:
        st.caption('Epoch đã train: ' + ' · '.join(f"{k}: {r['epochs']}" for k, r in attempt['trained'].items()))
    if attempt['version'] not in project['completed_rounds']:
        st.info('Vòng chưa có snapshot nhãn chung. Có thể xem dự đoán từng detector đã lưu; '
                'phần còn thiếu được đánh dấu rõ, không tự chạy lại model.')
    rows = []
    for phase, title in (('before', 'Trước train'), ('after', 'Sau train')):
        for key in ('A', 'B'):
            run = attempt['phases'][phase].get(key)
            rows.append({'Thời điểm': title, 'Detector': key,
                'Frame đã predict': f"{run['completed']}/{run['total']}" if run else 'Chưa có',
                'Kết quả': 'Đầy đủ' if run and run['complete'] else 'Chưa đầy đủ'})
    st.dataframe(rows, hide_index=True, width='stretch')
    frames = {f['frame_id']: f for f in read(store.path(project['id']) / 'dataset.json')['frames']}
    frame_id = st.selectbox('Frame sample dự đoán', project['sample_frame_ids'],
                            key='preview_frame_' + project['id'])
    chosen = st.multiselect('Class hiển thị', project['schema']['classes'],
                            default=attempt['adaptation_classes'], key='preview_classes_' + project['id'])
    score = st.slider('Score model tối thiểu', 0., 1., .05, .05, key='preview_score_' + project['id'])
    st.caption('Score gốc của detector; không phải độ chính xác. Chọn cùng frame để so sánh trước/sau. '
               'Point cloud được lấy từ dữ liệu đầu vào, không đọc nhãn gold.')
    frame = frames[frame_id]
    points, total_points = None, None
    try:
        stat = Path(frame['media']).stat()
        points, total_points = cloud(frame['media'], frame['sha256'], attempt['bin_columns'], stat.st_mtime_ns, stat.st_size)
    except (ImportError, OSError, ValueError) as error:
        st.warning('Chưa đọc được point cloud: ' + str(error) + '. Vẫn có thể xem hình học box.')
    if points is not None:
        st.caption(f'{total_points:,} điểm · hiển thị {len(points):,} điểm để thao tác mượt · BEV nhìn từ trên xuống')
    views = {}
    for phase in ('before', 'after'):
        for key in ('A', 'B'):
            run = attempt['phases'][phase].get(key)
            document = previews.prediction(run, frame_id) if run else None
            if document is None:
                views[phase, key] = None
                continue
            filtered = dict(document, boxes=[b for b in document['boxes']
                                            if b['class_name'] in chosen and b['score'] >= score])
            figure = box_plot(filtered, '3d')
            if points is not None:
                scatter = go.Scattergl(x=points[:, 0], y=points[:, 1], mode='markers', name='LiDAR',
                    marker=dict(size=1, color='#a2a7b3', opacity=.45), hoverinfo='skip')
                figure = go.Figure(data=[scatter, *figure.data], layout=figure.layout)
            views[phase, key] = (filtered, figure)
    # Identical axes avoid misleading changes caused by independent auto-zoom.
    xs, ys = [], []
    if points is not None:
        bounds = np.quantile(points[:, :2], [.01, .99], axis=0)
        xs.extend(bounds[:, 0]); ys.extend(bounds[:, 1])
    for view in views.values():
        if view:
            for trace in view[1].data:
                if trace.name != 'LiDAR':
                    xs.extend(trace.x); ys.extend(trace.y)
    if xs and ys:
        for view in views.values():
            if view:
                view[1].update_xaxes(range=[min(xs) - 2, max(xs) + 2])
                view[1].update_yaxes(range=[min(ys) - 2, max(ys) + 2], autorange=False)
    for phase, title in (('before', 'Trước train'), ('after', 'Sau train')):
        st.markdown('**' + title + ' ' + attempt['version'] + '**')
        for column, key, model in zip(st.columns(2), ('A', 'B'), ('SECOND-IoU', 'PV-RCNN')):
            with column:
                st.markdown('**' + key + ' · ' + model + '**')
                view = views[phase, key]
                if view is None:
                    st.info('Frame này chưa có prediction ở bước này.')
                    continue
                document, figure = view
                st.caption(f"{len(document['boxes'])} box theo bộ lọc đang chọn")
                st.plotly_chart(figure, width='stretch', key=f'prediction_{project["id"]}_{phase}_{key}')
                with st.expander('Chi tiết box & score'):
                    st.json(document)
