"""Local ensemble in the existing choose -> prepare -> run -> results workflow."""
from pathlib import Path
import json
import streamlit as st
from annoloop import local_ensemble as service
from annoloop.io import read, digest
from components import box_plot


def inventory_table(rows):
    st.dataframe([{'Checkpoint':m['name'], 'Kiến trúc':m['architecture'],
        'Class theo config':', '.join(m['classes']) or 'Chưa xác minh',
        'Nguồn':m['origin'], 'MB':round(m['bytes']/1e6, 1),
        'Trạng thái':'Có thể gửi kiểm tra/predict' if m['eligible'] else 'Chưa tương thích',
        'Lý do':' · '.join(m['reasons']), 'File':m['path']} for m in rows],
        hide_index=True, width='stretch')


def choose(store, settings, project):
    """Return True when the user is viewing the N-model selection route."""
    pid=project['id']; saved=service.configuration(store,pid)
    mode=st.radio('Cách sử dụng model', ['Adaptation A/B','Ghép checkpoint local'],
        index=int(bool(saved.get('enabled'))), horizontal=True, key='ensemble_mode_'+pid)
    rows=service.discover(store,pid,settings)
    imported=[m for m in rows if m['origin'].startswith('ST3D chính thức')]
    with st.expander(f'Checkpoint trên máy · {len(rows)} file · {len(imported)} ST3D mới tải',
                     expanded=mode=='Ghép checkpoint local'):
        if rows:inventory_table(rows)
        else:st.info('Chưa tìm thấy checkpoint có receipt/config trong thư mục dữ liệu của app.')
        st.caption('Đọc receipt/config local, không load model hoặc tải mạng. “Có thể gửi kiểm tra” chưa phải chứng nhận GPU/chất lượng. File cùng hash được gộp khi lưu lựa chọn.')
    if mode=='Adaptation A/B':
        if saved.get('enabled'):
            st.info('Workflow đang lưu là ghép local. Bấm dưới đây để quay về A/B; các lượt cũ được giữ.')
            if st.button('Dùng lại workflow A/B',key='disable_ensemble_'+pid):
                service.disable(store,pid);st.rerun()
            return True
        return False
    st.subheader('Nhiều checkpoint → một bộ nhãn')
    st.caption('Chạy lần lượt trên GPU, calibration source theo từng checkpoint/class rồi hợp nhất dự đoán. Đây là inference ensemble, không train thêm hoặc trộn trọng số .pth.')
    if project['schema']['data_type']!='3d':
        st.warning('Luồng ghép hiện mới có adapter 3D; detector/calibration 2D chưa được nối.')
        return True
    eligible={m['path']:m for m in rows if m['eligible']}
    pending=st.session_state.pop('ensemble_pending_'+pid,None)
    if pending is not None:st.session_state['ensemble_selected_'+pid]=pending
    defaults={} if 'ensemble_selected_'+pid in st.session_state else dict(
        default=[m['path'] for m in saved.get('models',[]) if m['path'] in eligible] if saved.get('models') else list(eligible))
    selected=st.multiselect('Checkpoint tham gia ghép',list(eligible),
        format_func=lambda path:eligible[path]['architecture']+' · '+Path(path).stem+' · '+eligible[path]['origin'],
        key='ensemble_selected_'+pid,**defaults)
    if st.button('Chọn tất cả checkpoint đủ điều kiện',disabled=not eligible,key='ensemble_all_'+pid):
        st.session_state['ensemble_pending_'+pid]=list(eligible);st.rerun()
    adaptation_path=store.path(pid)/'adaptation.json'
    adaptation=read(adaptation_path) if adaptation_path.exists() else {}
    classes=st.multiselect('Class cần ghép',project['schema']['classes'],
        default=saved.get('classes',adaptation.get('adaptation_classes',project['schema']['classes'])),
        key='ensemble_classes_'+pid)
    coverage={name:[m for path,m in eligible.items() if path in selected and name in m['classes']] for name in classes}
    st.dataframe([{'Class':name,'Checkpoint hỗ trợ':len(models),
        'Nhóm độc lập':len({m['vote_group'] for m in models})} for name,models in coverage.items()],hide_index=True,width='stretch')
    st.caption('Nhiều epoch của cùng kiến trúc chỉ tính một nhóm biểu quyết. Model không hỗ trợ một class không bị tính là bỏ sót class đó. Một nhóm duy nhất cần ngưỡng chặt hơn.')
    columns=st.selectbox('Số cột nếu dữ liệu BIN',[3,4,5],index=[3,4,5].index(saved.get('bin_columns',4)),
        format_func=lambda n:{3:'3 · XYZ',4:'4 · XYZI',5:'5 · XYZIT'}[n],key='ensemble_bin_'+pid)
    confirmed=st.checkbox('Tôi xác nhận XYZ theo mét; X trước, Y trái, Z lên',
        value=saved.get('coordinates_confirmed',False),key='ensemble_axes_'+pid)
    st.caption('Mặc định: IoU ≥ 0.5; confidence ≥ 0.7; quality ≥ 0.5; ít nhất 5 point. Một nhóm: confidence ≥ 0.9 và quality ≥ 0.6. Đây là ngưỡng chọn nhãn, không bảo đảm accuracy.')
    if st.button('Lưu nhóm model & tiếp tục →',type='primary',disabled=not selected or not classes or not confirmed,
                 key='save_ensemble_'+pid):
        service.save(store,pid,settings,selected,classes,confirmed,columns)
        st.session_state['pending_navigation']='Huấn luyện nguồn';st.rerun()
    return True


def prepare(store,settings,project):
    config=service.configuration(store,project['id'])
    if not config.get('enabled'):return False
    st.subheader('Nhóm checkpoint đã lưu')
    st.write(f"{len(config['models'])} checkpoint · {', '.join(config['classes'])}")
    inventory_table(config['models'])
    st.info('Không cần pretrain cho luồng ghép inference. Khi bấm chạy, từng model strict-load rồi predict source validation; đủ calibration mới predict target. Model lỗi sẽ dừng lượt và giữ artifacts.')
    if st.button('Tiếp tục: chạy ghép checkpoint →',type='primary'):
        st.session_state['pending_navigation']='Vòng adaptation';st.rerun()
    return True


@st.fragment(run_every=5)
def monitor(store,project):
    paths=service.jobs(store,project['id'])
    if not paths:return
    root=paths[0].parent;state=service.status(root)
    st.write('Lượt '+root.name[:8]+' · '+state['status'])
    st.caption('Bước: '+state.get('stage','')+' · '+str(state.get('model') or ''))
    if state.get('model_total'):st.progress(state.get('model_index',0)/state['model_total'],text='Model đang xử lý')
    if state.get('error'):st.error(state['error'])
    logs=[root/'worker.log']+sorted(root.glob('model_*/**/inference.log'),key=lambda p:p.stat().st_mtime)[-1:]
    with st.expander('Log thực tế'):
        for log in logs:
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-12000));st.code(stream.read().decode(errors='replace'),language=None)
    if state['process_alive'] and st.button('Dừng ghép · giữ artifacts',key='stop_ensemble_'+root.name):
        service.stop(root);st.rerun()
    if state['status']=='completed':
        st.success('Đã có kết quả. Mở bước 5 · Kết quả & CVAT.')


def execution(store,settings,project):
    config=service.configuration(store,project['id'])
    if not config.get('enabled'):return False
    st.subheader('Ghép dự đoán từ checkpoint local')
    st.write(f"{len(config['models'])} checkpoint · {', '.join(config['classes'])}")
    st.caption('Source validation → calibration → predict target cho từng model, chạy lần lượt. Sau đó ghép theo class/IoU và kiểm tra point support. Chỉ dùng split adaptation; không đọc gold target, không train.')
    running=service.active(store,project['id'])
    if st.button('Chạy kiểm tra & ghép dự đoán',type='primary',disabled=running):
        with st.spinner('Kiểm tra dữ liệu, runtime và GPU…'):
            service.start(store,project['id'],settings)
        st.rerun()
    monitor(store,project)
    return True


def results(store,settings,project):
    if not service.configuration(store,project['id']).get('enabled'):return False
    st.subheader('Kết quả ghép checkpoint')
    completed=[p.parent for p in service.jobs(store,project['id'])
               if service.status(p.parent)['status']=='completed' and (p.parent/'result.json').is_file()]
    if not completed:
        st.info('Chưa có lượt ghép hoàn tất. Chạy ở bước 4; frame chưa predict không được hiển thị như frame không có xe.')
        return True
    root=st.selectbox('Lượt ghép đã lưu',completed,format_func=lambda p:p.name[:12])
    result=read(root/'result.json');lookup={f['frame_id']:f for f in result['frames']}
    if st.button('Tiếp tục: final predict & CVAT →'):
        st.session_state['pending_navigation']='Xuất CVAT';st.rerun()
    st.write(f"{len(lookup)} frame adaptation · {result['positive_count']} box nhận · {result['ignored_count']} vùng ignore")
    st.caption('Các số đếm là thống kê đầu ra, không phải accuracy. Đây là shape theo frame; chưa tạo temporal track hoặc thay nhãn M đã chọn.')
    samples=[fid for fid in project['sample_frame_ids'] if fid in lookup]
    fid=st.selectbox('Frame xem kết quả',list(lookup),index=list(lookup).index(samples[0]) if samples else 0)
    doc=read(root/lookup[fid]['file'])
    include=st.checkbox('Hiện vùng ignore',value=False)
    view=dict(doc,boxes=doc['boxes']+(doc['ignored'] if include else []))
    figure=box_plot(view,'3d')
    from annoloop.inspection import load_xyz
    import plotly.graph_objects as go
    inventory=read(store.path(project['id'])/'dataset.json')
    frame=next(f for f in inventory['frames'] if f['frame_id']==fid)
    if digest(frame['media']) != frame['sha256']:raise ValueError('Media preview đã thay đổi')
    points=load_xyz(frame['media'],read(root/'spec.json')['config']['bin_columns'])
    points=points[::max(1,len(points)//12000)]
    figure.add_trace(go.Scattergl(x=points[:,0],y=points[:,1],mode='markers',
        marker=dict(size=1,color='#8893a7',opacity=.4),name='Point cloud'))
    st.plotly_chart(figure,width='stretch')
    with st.expander('Nguồn từng box / confidence / quality / số nhóm biểu quyết'):st.json(doc)
    st.download_button('Tải nhãn ghép JSON',json.dumps(doc,ensure_ascii=False,indent=2),file_name='ensemble_frame.json')
    package=Path(result['cvat_package'])
    if digest(package)!=result['cvat_sha256']:raise ValueError('Gói CVAT đã thay đổi')
    st.download_button('Tải gói shape cho exporter CVAT',package.read_bytes(),file_name='ensemble_cvat_shapes.json',mime='application/json')
    st.caption('Gói class-name cho exporter AnnoLoop; chưa phải ZIP import trực tiếp CVAT. Chưa tự upload hoặc đưa holdout vào bàn giao.')
    return True
