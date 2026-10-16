"""Architecture-neutral checkpoint discovery. Viewing never starts a download/GPU."""
import streamlit as st
from annoloop.model_hub import (load_catalog,recommendations,canonical,status,start_download,
                                cancel_download,model_root,installed)
from annoloop.io import read
from components import hero,chips


def size_label(value):
    return f'{value/1024**2:.1f} MiB' if value else 'chưa biết dung lượng'


@st.fragment(run_every='2s')
def download_progress(settings,model,namespace='',selected=False):
    state=status(settings.data,model['id']);mode=state['status']
    widget_id=model['id']+'_'+namespace
    labels={'not_installed':'Chưa tải','queued':'Đang mở worker','downloading':'Đang tải',
            'completed':'Đã tải','failed':'Tải lỗi · có thể thử lại','paused':'Đã dừng · giữ phần tải dở',
            'interrupted':'Worker bị ngắt · có thể tiếp tục','cache_invalid':'Cần tải lại'}
    st.markdown('**'+labels.get(mode,mode)+'**')
    total=state.get('total_bytes');done=state.get('bytes_downloaded',0)
    if total:st.progress(min(done/total,1.),text=size_label(done)+' / '+size_label(total))
    elif mode=='downloading':st.caption('Đã nhận '+size_label(done))
    if state.get('error'):st.error(state['error'])
    if mode in ('queued','downloading'):
        if st.button('Dừng tải, giữ phần đã tải',key='cancel_model_'+widget_id):
            cancel_download(settings.data,model['id']);st.info('Đã gửi yêu cầu dừng; worker sẽ dừng sau lần nhận dữ liệu đang chờ.')
    else:
        if st.button('Tải & chuẩn bị model' if selected else 'Chọn model & chuẩn bị adapter',key='prepare_model_'+widget_id,type='primary',width='stretch',disabled=model['modality']!='point_cloud'):
            try:
                start_download(settings,model['id'],prepare=True);st.rerun()
            except (ValueError,OSError) as error:st.error(str(error))
        title='Kiểm tra / dùng lại file đã tải' if mode=='completed' else 'Tải / tiếp tục tải gói model'
        if st.button(title,key='download_model_'+widget_id,width='stretch'):
            try:
                start_download(settings,model['id']);st.rerun()
            except (ValueError,OSError) as error:st.error(str(error))
    if mode=='completed':
        st.caption('Lưu tại');st.code(str(model_root(settings.data,model['id'])),language=None)
        st.caption('Weights + config theo đúng phiên bản đã pin. SHA256 local lưu trong installed.json; chưa phải chứng nhận tương thích GPU.')
    selection=model_root(settings.data,model['id'])/'selection.json'
    if selection.exists():
        s=read(selection)
        if s['status']=='prepared_unverified':
            st.success('Đã tạo adapter binding tự động từ framework của model.')
            st.caption('Chưa chạy kiểm tra GPU/forward/backward. Generic adapter hiện hỗ trợ raw predict; train adaptation vẫn cần đường loss/ignore/calibration phù hợp.')
        elif s['status']=='failed':st.error('Chuẩn bị adapter lỗi: '+s.get('error',''))
        else:st.caption('Chuẩn bị model: '+s['status'])
    connection=model_root(settings.data,model['id'])/'connection.json'
    if connection.exists():
        report=read(connection)
        st.caption('Kết nối sau tải: '+report['status']+' · adaptation chưa được cấp quyền chạy.')
        with st.expander('Các phần kết nối hiện có / còn thiếu'):
            st.json(report.get('capabilities',{}))
            for missing in report['missing_steps']:st.write('• '+missing)


def library(settings,classes=()):
    hero('KHO MODEL','Chọn nhãn. Tìm model. Tải khi cần.',
         'Danh mục nhiều kiến trúc từ model zoo chính thức. Class lấy từ config; link weights đã kiểm tra trả binary thật.')
    catalog=load_catalog();models=catalog['models']
    a,b,c=st.columns(3)
    a.metric('Checkpoint tải được',len(models));b.metric('Họ / biến thể detector',len({m.get('family',m['architecture']) for m in models}));c.metric('Dataset nguồn',len({m['dataset'] for m in models}))
    st.caption('Kiểm tra link gần nhất: '+catalog['generated_at'][:10]+'. Link có thể thay đổi; lỗi tải được ghi lại và không tạo file cài đặt thành công.')
    st.caption('Xem/chuyển lựa chọn không tải gì. Bấm “Chọn model & chuẩn bị adapter” mới tải gói được chọn và khởi tạo adapter theo framework.')
    available=sorted({canonical(c) for m in models for c in m['classes']}|set(classes))
    wanted=st.multiselect('Class bạn cần gán',available,default=list(classes),accept_new_options=True,key='hub_classes',
        help='Tool không đoán class từ point cloud. Vehicle không tự tách thành car/truck/bus; chỉ chuẩn hóa viết hoa và dấu phân cách.')
    a,b=st.columns(2)
    dataset=a.selectbox('Dataset pretrained',['Tất cả']+sorted({m['dataset'] for m in models}),key='hub_dataset')
    architecture=b.selectbox('Kiến trúc',['Tất cả']+sorted({m.get('family',m['architecture']) for m in models}),key='hub_architecture')
    modality=st.selectbox('Dữ liệu model yêu cầu',sorted({m['modality'] for m in models},key=lambda x:x!='point_cloud'),format_func=lambda x:{'point_cloud':'Point cloud','lidar_camera':'LiDAR + ảnh camera','camera':'Ảnh camera'}.get(x,x),key='hub_modality')
    only_full=st.checkbox('Chỉ hiện model hỗ trợ đủ class đã chọn',value=bool(wanted),key='hub_only_full')
    rows=recommendations(wanted,catalog=catalog,modality=modality)
    if dataset!='Tất cả':rows=[m for m in rows if m['dataset']==dataset]
    if architecture!='Tất cả':rows=[m for m in rows if m.get('family',m['architecture'])==architecture]
    if only_full and wanted:rows=[m for m in rows if m['all_classes_supported']]
    if not rows:
        st.info('Chưa có checkpoint đã xác minh hỗ trợ lựa chọn này. Bỏ lọc đủ class để xem model thiếu nhãn; không tự train chỉ vì chưa tìm thấy trong danh mục.')
        return
    st.caption('Ưu tiên độ phủ class và mức tương thích. Không so AP của các dataset khác nhau để xếp hạng chất lượng.')
    st.dataframe([{'Model':m['name'],'Kiến trúc':m['architecture'],'Dataset':m['dataset'],
        'Hỗ trợ':', '.join(m['covered_classes']) or '—','Thiếu':', '.join(m['missing_classes']) or '—',
        'Dung lượng':size_label(m['download_verification'].get('bytes')),
        'Adapter MVP':'Đã kiểm chứng' if m['runtime_verified'] else 'Cần adapter / kiểm tra'} for m in rows],hide_index=True,width='stretch')
    lookup={m['id']:m for m in rows}
    selected=st.selectbox('Xem và tải checkpoint',list(lookup),format_func=lambda key:lookup[key]['architecture']+' · '+lookup[key]['dataset']+' · '+lookup[key]['name'],key='hub_selected')
    m=lookup[selected]
    with st.container(border=True):
        st.subheader(m['architecture']+' · '+m['dataset'])
        st.caption(m['name']);chips(m['classes'])
        if wanted:
            (st.success if m['all_classes_supported'] else st.info)('Đủ '+str(len(wanted))+' class yêu cầu' if m['all_classes_supported'] else 'Thiếu: '+', '.join(m['missing_classes']))
        for note in m['mapping_notes']:st.info(note)
        if not m['runtime_verified']:st.warning(m['runtime_reason'])
        st.caption('Hỗ trợ class là thông tin taxonomy. Chất lượng trên dataset của bạn chưa được chấm.')
        a,b=st.columns(2)
        a.link_button('Nguồn chính thức / kết quả công bố',m['source_url'],width='stretch')
        b.link_button('Link tải weights trực tiếp',m['weights_url'],width='stretch')
        with st.expander('Config, tài nguyên và nguồn gốc'):
            st.write('Framework: '+m['framework']);st.write('Revision: '+m['revision'])
            st.write('Class: '+m['classes_evidence'])
            if m.get('training_memory_gb'):st.caption(f"VRAM train do upstream công bố: {m['training_memory_gb']} GB. Không phải yêu cầu inference hoặc bảo đảm chạy trên GPU của bạn.")
            st.write(m['license_note']);st.json(m.get('upstream_metrics',{}))
        download_progress(settings,m)
    st.info('Tải gói không đổi checkpoint của dự án, không mở trainer và không tự chạy GPU. Model mới chỉ vào luồng adaptation sau khi có adapter và kiểm tra tương thích thành công.')


def onboarding_hint(labels):
    rows=recommendations(labels);full=[m for m in rows if m['all_classes_supported']]
    with st.container(border=True):
        st.markdown('**Tìm checkpoint trên mạng theo class**')
        st.caption(f'{len(full)} gói trong danh mục hỗ trợ đủ lựa chọn. Nhiều kiến trúc, không giới hạn SECOND-IoU/PV-RCNN. Chưa tải thì chưa chiếm dung lượng weights.')
        if st.button('Mở kho model và đề xuất',key='open_model_hub'):
            st.session_state['hub_classes']=list(labels)
            st.session_state['pending_navigation']='Kho model';st.rerun()
