"""Post-download checklist. Only a deliberate submit can start verification."""
import streamlit as st
from annoloop import preparation
from annoloop.io import read
from annoloop import preparation_jobs
from components import tail, live_fragment


def monitor_preparation(root):
    paths=sorted((root/'preparation_jobs').glob('*/state.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    live=bool(st.session_state.get('preparation_waiting_'+root.name)) or bool(paths and preparation_jobs.status(paths[0].parent)['process_alive'])
    _monitor_preparation(root,live=live)


@live_fragment(3)
def _monitor_preparation(root):
    paths=sorted((root/'preparation_jobs').glob('*/state.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not paths:return
    job=paths[0].parent;state=preparation_jobs.status(job);waiting='preparation_waiting_'+root.name
    if st.session_state.get(waiting) and not state['process_alive']:
        st.session_state.pop(waiting,None);st.rerun(scope='app')
    if state['process_alive']:st.session_state[waiting]=True
    st.write('Chuẩn bị model: **'+state['status']+'** · '+state.get('stage',''))
    if state.get('model_id'):st.caption(state['model_id'])
    if state.get('error'):st.error(state['error'])
    with st.expander('Log chuẩn bị model'):st.code(tail(job/'worker.log'),language='log')
    if st.button('Dừng chuẩn bị, giữ file và log',key='stop_preparation_'+root.name,disabled=not state['process_alive']):
        preparation_jobs.stop(job);st.rerun()


def panel(store,settings,project):
    root=store.path(project['id']);path=root/'adaptation.json'
    if not path.exists():
        st.info('Chọn model ở bước 2 trước khi kiểm chứng adaptation.');return
    config=read(path)
    report=preparation.overview(store,project['id'],fast=True)
    st.subheader('Kiểm chứng sau khi tải / chuẩn bị model')
    st.info('Chế độ đã lưu: '+('Scan hiện tại · Δt=0, không có lịch sử' if config.get('mmdet_input_mode')=='current_scan_only' else 'Yêu cầu đầu vào gốc · strict')+'. Thay đổi trong form cần bấm Lưu hoặc nút kiểm tra để áp dụng.')
    st.caption('Chọn class & input → tải → nối recipe → predict / backward / ignore / calibration → mở adaptation khi đạt. Chưa chạy kiểm tra không được coi là đã đạt.')
    st.caption('Recipe đã có dùng batch nhỏ kiểm tra GPU, sau đó predict toàn bộ source validation để fit calibration. MMDetection3D dùng IoU đo độ ổn định qua phép quay, không phải IoU head. Không dùng nhãn target, không cập nhật weights; calibration có thể mất thời gian.')
    if any(m.get('framework')=='mmdetection3d' for m in config.get('models',{}).values()):
        with st.expander('Môi trường MMDetection3D · chuẩn bị một lần'):
            st.write('App dùng venv UI riêng. Worker cần MMDetection3D1.4 và các extension tương thích; runtime được kiểm tra khi bạn bấm nút chung. Chưa cài thì chạy script này trong terminal:')
            st.code('bash buildphase/annoloop_mvp/scripts/setup_mmdet3d.sh',language='bash')
            st.caption('Script tạo .venv-mmdet3d riêng; không tự chạy khi mở trang. Checkpoint mới thuộc recipe đã hỗ trợ dùng lại runtime này, không cần tạo JSON/YAML.')
    labels={'passed':'Đạt','pending':'Chờ kiểm tra','missing':'Thiếu code recipe','failed':'Chưa đạt'}
    st.dataframe([{'Detector':r['role'],'Model':r['model'],'Bước':r['stage'],
        'Trạng thái':labels[r['status']],'Chi tiết':r['detail']} for r in report['stages']],hide_index=True,width='stretch')
    sweep_only=[b for b in report['blockers'] if any(t in b for t in ('sweep','time thật','thiếu feature: time'))]
    if sweep_only and config.get('mmdet_input_mode','strict')=='strict':
        st.warning('Model đòi multi-sweep và cột time, nhưng dữ liệu chỉ có scan đơn. Nếu mỗi file là một scan riêng '
                   '(như KITTI XYZI), chọn **"Mỗi file là scan hiện tại"** trong form dưới rồi bấm kiểm tra. '
                   'Tool không tự đổi chế độ này cho bạn.')
    with st.form('prepare_checks_'+project['id']):
        axes=st.checkbox('Target dùng mét; X trước, Y trái, Z lên',value=config.get('coordinates_confirmed',False))
        columns=st.selectbox('BIN · số cột cho bước kiểm tra',[3,4,5],index=[3,4,5].index(config.get('bin_columns',4)),
            help='KITTI là 4 cột XYZI; PCD đọc header. Không tự tạo intensity. Elapsed time chỉ suy ra trong chế độ scan hiện tại đã chọn.')
        input_modes=['strict','current_scan_only']
        input_mode=st.radio('Đầu vào MMDetection3D',input_modes,index=input_modes.index(config.get('mmdet_input_mode','strict')),
            format_func=lambda v:'Giữ yêu cầu đầu vào gốc' if v=='strict' else 'Mỗi file là scan hiện tại · không dùng sweep lịch sử',
            help='Chỉ chế độ thứ hai mới suy ra elapsed time Δt=0 so với scan tham chiếu hiện tại. Không tạo sweep, không đổi encoder/weights, không tự tạo intensity. Có thể giảm chất lượng so với nhiều sweep.')
        save=st.form_submit_button('Lưu chế độ đầu vào',disabled=report['busy'])
        submit=st.form_submit_button('Tải, chuẩn bị adapter & kiểm tra model',type='primary',disabled=report['busy'])
    if save:
        try:
            preparation.save_input_setup(store,project['id'],settings,axes,columns,input_mode)
            st.rerun()
        except (ValueError,OSError,RuntimeError,KeyError) as error:st.error(str(error))
    if submit:
        try:
            with st.spinner('Kiểm tra file và recipe của các model đã chọn…'):
                preparation_jobs.start(store,project['id'],settings,axes,columns,input_mode)
            st.rerun()
        except (ValueError,OSError,RuntimeError,KeyError) as error:st.error(str(error))
    if report['ready']:
        st.success('Đã đạt kiểm chứng trên checkpoint/source/target hiện tại. Có thể mở adaptation.')
    elif any(r['status']=='missing' for r in report['stages']):
        st.info('Checkpoint vẫn được giữ. Cần triển khai recipe còn thiếu; tải xong không có nghĩa đã chạy được adaptation.')
    with st.expander('Lý do chưa mở adaptation'):
        for reason in report['blockers']:st.write('• '+reason)
    if st.button('Mở vòng adaptation →',disabled=not report['ready'],key='open_verified_adaptation_'+project['id']):
        st.session_state['pending_navigation']='Vòng adaptation';st.rerun()
    from adaptation_page import monitor
    monitor_preparation(root)
    monitor(root)
