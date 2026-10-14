"""Three-step folder wizard; no JSON/YAML entry in the default workflow."""
from pathlib import Path
import streamlit as st
from annoloop.io import read
from annoloop.schema import ClassSchema
from annoloop.selection import SelectionPolicy
from annoloop.settings import WORKSPACE
from annoloop.setup import inspect_source,inspect_target,model_catalog,make_plan,prepare_project
from components import hero,chips


@st.dialog('Chọn thư mục',width='large')
def folder_dialog(widget_key):
    cursor_key='browse_'+widget_key
    current=Path(st.session_state.get(cursor_key,st.session_state.get(widget_key) or WORKSPACE)).expanduser()
    if not current.is_dir():current=WORKSPACE
    st.code(str(current),language=None)
    if st.button('↑ Thư mục cha',disabled=current.parent==current):
        st.session_state[cursor_key]=str(current.parent);st.rerun(scope='fragment')
    try:
        directories=sorted([p for p in current.iterdir() if p.is_dir() and not p.name.startswith('.')],key=lambda p:p.name.lower())
    except OSError:
        st.error('Không đọc được thư mục này. Hãy chọn thư mục cha.');return
    selected=st.selectbox('Thư mục con',[None]+[str(p) for p in directories],format_func=lambda p:Path(p).name if p else 'Chọn thư mục con',placeholder='Chọn thư mục con')
    a,b=st.columns(2)
    if a.button('Mở thư mục con',disabled=not selected):
        st.session_state[cursor_key]=selected;st.rerun(scope='fragment')
    if b.button('Dùng thư mục này',type='primary'):
        st.session_state[widget_key]=str(current)
        st.rerun()


def folder_field(label,key):
    a,b=st.columns([5,1],vertical_alignment='bottom')
    with a:st.text_input(label,key=key,placeholder='Chọn thư mục trên máy chạy AnnoLoop')
    with b:
        if st.button('📂 Chọn',key=key+'_browse',width='stretch'):folder_dialog(key)
    return st.session_state.get(key,'')


def wizard(store,settings):
    hero('THIẾT LẬP NHANH','Chọn dữ liệu. Chọn nhãn. Tool lo phần còn lại.', 'Schema, inventory và cấu hình model được tạo tự động. Chỉ học nguồn khi checkpoint còn thiếu class yêu cầu.')
    st.subheader('1. Source và target')
    source_path=folder_field('Source · dataset có nhãn','quick_source')
    target_path=folder_field('Target · dataset cần gán nhãn','quick_target')
    st.caption('Source: nuScenes đã chuẩn bị · Target: point cloud PCD/NPY/BIN. Tool tự nhận cấu trúc thư mục và nhãn nguồn.')
    signature=(source_path,target_path)
    if st.session_state.get('quick_signature')!=signature:
        st.session_state.pop('quick_data',None)
    if st.button('Đọc hai thư mục',type='primary',disabled=not(source_path and target_path)):
        st.session_state.pop('quick_data',None)
        with st.spinner('Nhận dạng dataset và đọc nhãn nguồn…'):
            source=inspect_source(source_path);target=inspect_target(target_path)
            st.session_state['quick_data']=(source,target)
            st.session_state['quick_signature']=signature
    data=st.session_state.get('quick_data')
    if not data:return
    source,target=data
    a,b,c=st.columns(3)
    a.metric('Source train',len(source['frames']['train']));b.metric('Source validation',len(source['frames']['validation']));c.metric('Target media',target['frame_count'])
    if target['data_type']!='3d':
        st.info('Đã nhận dạng dataset 2D. Workflow hai detector 3D chưa áp dụng cho dữ liệu này.');return
    st.subheader('2. Nhãn cần gán')
    options=list(dict.fromkeys(target['classes']+source['schema']['classes']))
    labels=st.multiselect('Chọn nhãn',options,default=target['classes'],accept_new_options=True,
                         key='quick_labels_'+source['root']+target['root'],help='Nếu target chưa có schema, chọn các nhãn cần gán từ source hoặc nhập tên nhãn. Đây là yêu cầu gán nhãn, không phải nhãn gold target.')
    if not labels:
        st.info('Chọn ít nhất một nhãn để kiểm tra model.');return
    models=model_catalog(settings.data)
    st.subheader('3. Checkpoint pretrained')
    model_choices={};metadata={}
    for col,key,arch,title in zip(st.columns(2),['A','B'],['SECONDNetIoU','PVRCNN'],['A · SECOND-IoU','B · PV-RCNN']):
        choices=[m for m in models if m['architecture']==arch]
        lookup={m['metadata']:m for m in choices}
        with col:
            selected=st.selectbox(title,[m['metadata'] for m in choices]+[None],format_func=lambda p, entries=lookup:entries[p]['name'] if p else 'Không có checkpoint · học mới',key='quick_model_'+key)
            model_choices[key]=selected
            metadata[key]=read(selected) if selected else None
            if selected:chips(metadata[key]['classes'])
            else:st.caption('Model sẽ học từ source nếu source có nhãn cần thiết.')
    plan=make_plan(labels,source,metadata)
    rows=[]
    for name in labels:
        row={'Nhãn':name,'Source train':source['counts']['train'].get(name,0),'Source validation':source['counts']['validation'].get(name,0)}
        for key in ('A','B'):row['Model '+key]='Đã hỗ trợ' if name not in plan['models'][key]['missing_classes'] else 'Cần học'
        rows.append(row)
    st.dataframe(rows,width='stretch',hide_index=True)
    for key,item in plan['models'].items():
        (st.info if item['pretrain_required'] else st.success)(key+' · '+item['reason'])
    if plan['source_blockers']:
        st.warning(' · '.join(plan['source_blockers'])+'. Có thể lưu thiết lập, nhưng chưa thể train/calibration các class này.')
    with st.expander('Tùy chọn',expanded=False):
        name=st.text_input('Tên dự án',value=Path(target['root']).name)
        a,b=st.columns(2)
        rounds=a.number_input('Số vòng adaptation tối đa',min_value=1,value=3)
        execution=b.selectbox('Cách chạy adaptation',['manual','automatic'],format_func=lambda x:'Thủ công từng vòng' if x=='manual' else 'Tự động theo số vòng')
        selection=st.selectbox('Chọn phiên bản nhãn',['manual','automatic'],format_func=lambda x:'Tôi xem sample và chọn' if x=='manual' else 'Tự chọn theo ngưỡng')
        if any(item['pretrain_required'] for item in plan['models'].values()):
            epochs=st.number_input('Epoch học nguồn khi thiếu class',min_value=1,value=20)
            ros=st.checkbox('ROS · thay đổi kích thước object khi học nguồn',value=True)
        else:epochs=20;ros=True
        st.caption('Hai detector chạy lần lượt; checkpoint mỗi 10 epoch + cuối. Ngưỡng chọn nhãn có giá trị mặc định, xem ở màn Vòng adaptation.')
    st.caption('Bước dưới chỉ chuẩn bị dự án; chưa khởi chạy model hoặc trainer.')
    if st.button('Tạo dự án và tự chuẩn bị',type='primary',width='stretch'):
        with st.spinner('Kiểm tra checkpoint, tạo schema và cấu hình…'):
            pid=prepare_project(store,settings,name,source,target,labels,model_choices,int(rounds),execution,SelectionPolicy(mode=selection),int(epochs),(.9,1.1) if ros else None)
        st.session_state['pending_project_id']=pid
        st.session_state['pending_navigation']='Huấn luyện nguồn'
        st.session_state['flash']='Đã chuẩn bị tự động. Xem kế hoạch model bên dưới.'
        st.rerun()
