"""Final prediction -> immutable media/annotations -> explicit CVAT upload."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import importlib.util
import streamlit as st
from annoloop import final_delivery as delivery,local_ensemble
from annoloop.io import read
from annoloop.cvat import check_login
from result_viewer import render as render_result


def sdk_available():
    return importlib.util.find_spec('cvat_sdk') is not None


@st.fragment(run_every=5)
def monitor(root):
    state=local_ensemble.status(root)
    st.write('Final predict · '+state['status'])
    st.caption('Bước: '+state.get('stage','')+' · '+str(state.get('model') or ''))
    if state.get('model_total'):
        st.progress(state.get('model_index',0)/state['model_total'],text='Model đang xử lý')
    if state.get('frame_total'):
        st.caption(f"Đã ghép {state['frame_index']}/{state['frame_total']} frame")
    if state.get('error'):st.error(state['error'])
    paths=[root/'worker.log']+sorted(root.glob('model_*/**/inference.log'),key=lambda p:p.stat().st_mtime)[-1:]
    with st.expander('Log final predict'):
        for path in paths:
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(max(0,path.stat().st_size-10000))
                    st.code(stream.read().decode(errors='replace'),language=None)
    if state['process_alive']:
        if st.button('Dừng final predict · giữ artifacts',key='stop_delivery_'+root.name):
            local_ensemble.stop(root);st.rerun()
    elif state['status']=='completed':
        if st.button('Mở gói bàn giao đã tạo',key='open_delivery_'+root.name):st.rerun(scope='app')


def page(store,settings,project):
    pid=project['id']
    st.subheader('1 · Dự đoán cuối từ phiên bản đã chọn')
    st.caption('Chạy lại inference bằng checkpoint của vòng đang chọn hoặc của lượt ghép local hoàn tất. Không dùng snapshot pseudo-label làm kết quả final và không train thêm.')
    scope=st.selectbox('Phạm vi dữ liệu bàn giao',['non_holdout','delivery','all'],
        format_func=lambda v:{'non_holdout':'Toàn bộ adaptation + delivery · giữ riêng holdout',
            'delivery':'Chỉ split delivery','all':'Toàn bộ dataset · gồm cả holdout'}[v],key='delivery_scope_'+pid)
    if scope=='all':
        st.info('Holdout chỉ được suy luận để bàn giao ở bước này. Kết quả final không đưa ngược vào pseudo-label/train hoặc dùng chọn model.')
    chosen=None
    if local_ensemble.configuration(store,pid).get('enabled'):
        candidates=[p.parent for p in local_ensemble.jobs(store,pid)
                    if local_ensemble.status(p.parent)['status']=='completed' and (p.parent/'result.json').exists()]
        if candidates:chosen=st.selectbox('Lượt ghép checkpoint dùng để final predict',candidates,format_func=lambda p:p.name[:12])
    ready=False
    try:
        spec=delivery.plan(store,pid,scope,chosen)
        st.write('Phiên bản: **'+spec['version']+'**')
        st.write(f"{len(spec['frame_ids'])} frame · {len(spec['config']['models'])} model · "+', '.join(spec['config']['classes']))
        st.caption('Checkpoint, config và source validation sẽ được kiểm tra hash trước khi chạy. Mỗi model calibration riêng rồi dự đoán lần lượt trên GPU.')
        ready=True
    except (ValueError,KeyError,OSError) as error:
        st.info('Chưa thể final predict: '+str(error))
    running=delivery.active(store,pid)
    if st.button('Chạy final predict & tạo gói bàn giao',type='primary',disabled=not ready or running):
        with st.spinner('Kiểm tra checkpoint, runtime và GPU…'):
            delivery.start(store,pid,settings,scope,chosen)
        st.rerun()
    paths=delivery.jobs(store,pid)
    if paths:monitor(paths[0].parent)
    completed=[p.parent for p in paths if local_ensemble.status(p.parent)['status']=='completed']
    root=None;directory=None;document=None
    st.subheader('2 · Kiểm tra media và nhãn bàn giao')
    if completed:
        root=st.selectbox('Gói final đã hoàn tất',completed,format_func=lambda p:read(p/'result.json')['version']+' · '+p.name[:8])
        directory,document=delivery.preview_export(root)
        result=read(root/'result.json')
        st.success(f"Đã chuẩn bị {len(document['frame_ids'])} frame · {len(document['shapes'])} cuboid shape")
        st.caption('Phiên bản của gói: '+result['version']+' · phạm vi: '+result['scope']+'. Gói cũ giữ nguyên khi đổi vòng được chọn.')
        st.caption('Point cloud được chuyển thành PCD XYZ theo thứ tự frame cố định. Nhãn và media có SHA256. Vùng ignore không xuất thành nhãn dương.')
        render_result(store,project,root,result,'delivery',document['schema']['classes'])
        for name,label in [('annoloop_delivery.zip','Tải gói bàn giao AnnoLoop'),('media.zip','Tải point cloud PCD'),('annotations.json','Tải nhãn shape JSON')]:
            path=directory/name
            if path.stat().st_size<=64*1024*1024:
                st.download_button(label,path.read_bytes(),file_name=name,key='delivery_download_'+name)
            else:st.caption(label+' · file lớn, lấy tại: '+str(path))
        st.caption('ZIP AnnoLoop gồm media và nhãn cho uploader bên dưới; không phải định dạng ZIP annotation để import trực tiếp trong CVAT.')
    else:
        st.info('Chưa có gói final hoàn tất. Bấm chạy phía trên; các lượt bị ngắt giữ nguyên log và file đã tạo.')
    st.subheader('3 · Tạo task mới trên CVAT local')
    st.caption('Có thể kiểm tra đăng nhập ngay, kể cả khi chưa có gói final. Nút kiểm tra chỉ xác thực tài khoản, không tạo task.')
    st.caption('Hiện bàn giao dạng shape. Track cần timestamp/sequence và pose hoặc cảm biến cố định đã được xác nhận; không tự tạo track từ các frame chưa có thông tin này.')
    with st.form('delivery_upload_'+pid):
        host=st.text_input('Địa chỉ CVAT local',value='http://localhost:8080',key='delivery_host_'+pid)
        username=st.text_input('Tài khoản CVAT',key='delivery_user_'+pid)
        password=st.text_input('Mật khẩu CVAT',type='password',key='delivery_password_'+pid)
        name=st.text_input('Tên task CVAT',value=project['name']+' · final shape',key='delivery_task_'+pid)
        sdk=sdk_available()
        if not sdk:st.info('Venv UI chưa có cvat-sdk. Cài extra cvat để mở nút upload; vẫn tải được gói offline.')
        check=st.form_submit_button('Kiểm tra đăng nhập CVAT',disabled=not sdk)
        submitted=st.form_submit_button('Tạo task CVAT',type='primary',disabled=root is None or not sdk)
    login_key='delivery_login_'+pid
    if check:
        try:
            with st.spinner('Đăng nhập và xác minh tài khoản CVAT…'):
                account=check_login(host,username,password)
            st.session_state[login_key]=dict(ok=True,**account)
        except ValueError as error:
            st.session_state[login_key]=dict(ok=False,host=host.strip().rstrip('/'),
                                             username=username.strip(),error=str(error))
    account=st.session_state.get(login_key)
    if account and account['host']==host.strip().rstrip('/') and account['username']==username.strip():
        if account['ok']:
            st.success('Đã kiểm tra đăng nhập thành công: '+account['username'])
            checked=datetime.fromisoformat(account['checked_at']).astimezone(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%d/%m/%Y %H:%M:%S')
            st.caption('Lần kiểm tra: '+checked+' (giờ Việt Nam). Tạo task là thao tác riêng.')
        else:st.error(account['error'])
    if submitted:
        try:
            with st.spinner('Đưa media và nhãn lên CVAT, kiểm tra lại thứ tự frame và cuboid…'):
                receipt=delivery.upload(root,host,username,password,name)
            st.success('Task đã được kiểm tra sau upload.')
            st.link_button('Mở task CVAT',receipt['url'])
        finally:
            # Streamlit does not allow mutating a rendered widget; clear on next run.
            st.session_state['clear_delivery_password']=pid
    if directory and (directory/'upload.json').exists():
        receipt=read(directory/'upload.json')
        if receipt.get('status')=='verified':st.link_button('Task CVAT đã tạo',receipt['url'])
