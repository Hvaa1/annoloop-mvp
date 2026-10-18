"""Prepare only explicitly selected, pinned ST3D-related initializers."""
import streamlit as st
from annoloop.io import read
from annoloop.native_sources import entry,root_for,start_download
from annoloop.native_source_binding import bind
from annoloop.reused.training_monitor import process_identity
from components import live_fragment


def progress(settings,identifier):
    root=root_for(settings.data,identifier)
    launch=read(root/'launch.json') if (root/'launch.json').exists() else {}
    _progress(settings,identifier,live=bool(launch.get('process_start') and process_identity(launch.get('pid'))==launch['process_start']))


@live_fragment('2s')
def _progress(settings,identifier):
    root=root_for(settings.data,identifier)
    state=read(root/'download.json') if (root/'download.json').exists() else {}
    launch=read(root/'launch.json') if (root/'launch.json').exists() else {}
    alive=bool(launch.get('process_start') and process_identity(launch.get('pid'))==launch['process_start'])
    if state.get('status')=='completed':
        st.success('Đã tải/import và xác minh hash. Chưa kiểm chứng GPU.')
        if st.button('Kiểm tra / tải lại file native',key='recheck_native_'+identifier):
            start_download(settings,identifier);st.rerun()
        return
    if alive:
        st.caption('Đang tải checkpoint đã chọn; không tải toàn bộ archive.')
        if state.get('total_bytes'):st.progress(min(state.get('bytes_downloaded',0)/state['total_bytes'],1.))
        if st.button('Dừng tải native source',key='stop_native_'+identifier):
            (root/'cancel').touch();st.info('Đã gửi yêu cầu dừng.')
    else:
        if state.get('error'):st.warning(state['error'])
        if state.get('status')=='downloading':st.warning('Worker trước đã dừng; artifacts được giữ. Bấm để thử lượt tải mới.')
        if st.button('Tải pretrained ST3D/LSF',key='download_native_'+identifier,type='primary'):
            start_download(settings,identifier);st.rerun()


def prepare_native_sources(store,settings,project,setup):
    path=store.path(project['id'])/'adaptation.json'
    selection=read(path).get('model_selection',{}) if path.exists() else {}
    waiting=False
    for key,value in selection.items():
        identifier=value.get('source_id') if value.get('kind')=='source' else None
        if not identifier:continue
        model=entry(identifier)
        if setup.get('native_source_bindings',{}).get(key,{}).get('id')==identifier:continue
        waiting=True
        with st.container(border=True):
            st.subheader('Model '+key+' · '+model['name'])
            st.caption('Class gốc Vehicle · XYZ · source-only. Thiếu class mới học nguồn; không chạy train khi tải xong.')
            progress(settings,identifier)
            root=root_for(settings.data,identifier)
            if st.button('Kết nối initializer '+key+' vào dự án',key='bind_native_'+project['id']+'_'+key,
                         disabled=not (root/'checkpoint.metadata.json').exists()):
                bind(store,project['id'],settings,key,identifier);st.rerun()
    if waiting:
        st.info('Sau khi kết nối đủ initializer đã chọn, cấu hình nguồn và các nút train hiện ở đây. Các bước GPU/calibration vẫn kiểm tra riêng.')
        if st.button('← Quay lại chọn model',key='native_back'):
            st.session_state['pending_navigation']='Class & nguồn';st.rerun()
    return waiting
