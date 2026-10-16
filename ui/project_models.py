"""The single project model-selection step; all later pages consume this binding."""
from pathlib import Path
import streamlit as st
from annoloop import adaptation
from annoloop.io import read
from annoloop.model_hub import load_catalog,canonical,model_root
from annoloop.model_filter import dataset_profile,assess
from annoloop.native_sources import catalog as native_catalog
from components import chips


def selection_step(store,settings,project,setup):
    pid=project['id'];path=store.path(pid)/'adaptation.json'
    config=read(path) if path.exists() else {};saved=config.get('model_selection',{})
    choices=adaptation.model_choices(store,pid,settings)
    wanted=config.get('adaptation_classes',project['schema']['classes'])
    locked=bool(project['completed_rounds'] or adaptation.active_job(store,pid))
    st.subheader('Hai model cho toàn bộ workflow')
    st.caption('Chọn một lần tại đây. Bước Chuẩn bị model tải/học khi cần; Adaptation dùng đúng lựa chọn đã lưu.')
    inventory=read(store.path(pid)/'dataset.json')
    profile=dataset_profile([f for f in inventory['frames'] if f['split']=='adaptation'])
    with st.container(border=True):
        st.markdown('**Kiểm tra đầu vào trước khi chọn model**')
        chips(profile['features'] or ['Chưa xác định feature'])
        st.caption(f"Đã đọc header {profile['checked']}/{profile['total']} frame mẫu; không chạy detector, không đọc nhãn target. RGB không thay cho intensity.")
        if profile['mixed']:st.warning('Các frame mẫu có feature khác nhau. Bộ lọc dùng những feature có ở tất cả mẫu đọc được.')
        if profile['errors']:
            st.warning('Chưa xác định đủ feature đầu vào. Không tự coi model là tương thích.')
            with st.expander('Chi tiết frame chưa đọc được'):st.dataframe(profile['errors'],hide_index=True,width='stretch')
        st.caption('Dataset hiện chưa có manifest multi-sweep/timestamp/ego pose được adapter hỗ trợ.')
    compatible=st.checkbox('Chỉ hiện model phù hợp cho adaptation',value=True,key='workflow_compatible_'+pid,
        help='Kiểm tra class, feature đầu vào và backend train/calibration. Vẫn phải kiểm tra GPU sau khi chọn.')
    if not compatible:st.warning('Chế độ xem mở rộng: có thể chọn để chuẩn bị model, nhưng model không tương thích vẫn không chạy được adaptation.')
    full=st.checkbox('Chỉ hiện checkpoint đủ class',value=True,key='workflow_complete_'+pid)
    registry=load_catalog()['models'];selected={};all_selected=True;excluded={}
    native_registry=native_catalog()
    with st.expander('Nguồn ST3D / ST3D++ và các checkpoint liên quan'):
        st.link_button('ST3D chính thức · Model Zoo','https://github.com/CVMI-Lab/ST3D#model-zoo')
        st.caption('ST3D có SECOND-IoU, PV-RCNN, PointRCNN. Link repo không tự chứng nhận checkpoint/class hoặc khả năng chạy. PointRCNN chưa nối đủ trong MVP.')
        st.caption('Hai lựa chọn 3DOD-LSF bên dưới có payload SHA256 đã ghi nhận, class gốc Vehicle và đầu vào XYZ. Link archive sẽ được kiểm tra khi bạn bấm tải; không tải lúc xem trang.')
    for col,key in zip(st.columns(2),['A','B']):
        with col,st.container(border=True):
            options={};labels={};metadata={};reports={}
            current=saved.get(key)
            if current is None and key in config.get('models',{}):
                m=config['models'][key];current=dict(kind='hub',id=m['model_id']) if m.get('backend')=='registry_framework' else dict(kind='local',path=m['path'])
            for m in choices[key]:
                if m.get('backend','native_st3d')!='native_st3d':continue
                report=assess(dict(m,backend='native_st3d'),wanted,profile,setup['source_counts'])
                token='local:'+m['path'];value=dict(kind='local',path=m['path'])
                if not report['eligible']:excluded[token]={'Model':m['name'],'Lý do chưa phù hợp':' · '.join(report['reasons'])}
                if compatible and not report['eligible'] and current!=value:continue
                options[token]=value;labels[token]=m['name']+' · local';metadata[token]=m;reports[token]=report
            for m in registry:
                report=assess(m,wanted,profile,setup['source_counts']);enough=not report['missing_classes']
                if not report['eligible']:excluded[m['id']]={'Model':m.get('family',m['architecture'])+' · '+m['dataset']+' · '+m['name'],'Lý do chưa phù hợp':' · '.join(report['reasons'])}
                is_current=current==dict(kind='hub',id=m['id'])
                if (m.get('modality')!='point_cloud' or compatible and not report['eligible'] or full and not enough) and not is_current:continue
                token='hub:'+m['id'];options[token]=dict(kind='hub',id=m['id'])
                cached=(model_root(settings.data,m['id'])/'installed.json').exists()
                labels[token]=m.get('family',m['architecture'])+' · '+m['dataset']+' · '+m['name']+(' · đã tải' if cached else ' · chưa tải')+' · chưa có adaptation train'
                metadata[token]=m;reports[token]=report
            if setup['plan']['models'][key]['pretrain_required']:
                token='source';options[token]=dict(kind='source');labels[token]='Học class còn thiếu từ source · '+adaptation.ARCHITECTURES[key]
            for m in native_registry['models']:
                if m['architecture']!=adaptation.ARCHITECTURES[key]:continue
                missing=[c for c in wanted if c not in m['classes']]
                issues=[]
                if not profile['complete'] or not set(m['features'])<=set(profile['features']):issues.append('Chưa xác minh đầu vào XYZ')
                for split in ('train','validation'):
                    if any(setup['source_counts'][split].get(c,0)<=0 for c in wanted):issues.append('Source '+split+' thiếu class yêu cầu')
                value=dict(kind='source',source_id=m['id'])
                if compatible and issues and current!=value:
                    excluded[m['id']]={'Model':m['name'],'Lý do chưa phù hợp':' · '.join(issues)};continue
                token='source:'+m['id'];options[token]=value;metadata[token]=m
                labels[token]=m['name']+(' · học nhãn mới: '+', '.join(missing) if missing else ' · nối native, cần kiểm tra GPU')
            if compatible and not any(r['eligible'] for r in reports.values()):
                st.info('Chưa có checkpoint qua bộ lọc cho vai trò này. Xem lý do bên dưới; tool không tự train hoặc đổi model.')
            values=[None]+list(options);initial=next((v for v in options if options[v]==current),None)
            pick=st.selectbox('Model '+key,values,index=values.index(initial),format_func=lambda v,lookup=labels:lookup[v] if v else 'Chọn model',key=f'workflow_model_{pid}_{key}',disabled=locked)
            if pick is None:all_selected=False;continue
            selected[key]=options[pick]
            if options[pick]['kind']=='source':
                if pick!='source':
                    m=metadata[pick];chips(m['classes']);st.link_button('Nguồn checkpoint',m['source_url'])
                    missing=[c for c in wanted if c not in m['classes']]
                    if missing:st.info('Class gốc: Vehicle. Cần học '+', '.join(missing)+' bằng nhãn source; giữ backbone tương thích, khởi tạo lại head. Không tự đổi Vehicle thành car.')
                st.caption('Chỉ học nguồn khi thiếu class. Không tải hoặc chạy trainer tại bước này.')
            else:
                m=metadata[pick];chips(m['classes'])
                report=reports[pick]
                if report['eligible']:st.caption('Qua bộ lọc đầu vào/backend; chưa phải chứng nhận GPU hay chất lượng dự đoán.')
                else:
                    st.warning('Chưa phù hợp: '+' · '.join(report['reasons']))
                    if compatible:all_selected=False
                missing=[c for c in wanted if canonical(c) not in {canonical(n) for n in m['classes']}]
                (st.warning if missing else st.success)('Thiếu: '+', '.join(missing) if missing else 'Có đủ class yêu cầu')
                if pick.startswith('hub:'):
                    st.caption('Nguồn: '+m['framework'])
                    st.info('Gói framework này chưa có đầy đủ target training/calibration trong MVP. Có thể chọn để chuẩn bị, chưa thể chạy adaptation.')
    if excluded:
        with st.expander(f"{len(excluded)} model chưa phù hợp · xem lý do"):
            st.dataframe(list(excluded.values()),width='stretch',hide_index=True)
            st.caption('Thiếu feature: dùng checkpoint đúng đầu vào hoặc cung cấp feature thật. Thiếu adapter: chờ backend hỗ trợ. Không khắc phục hai trường hợp này bằng train class mới.')
    if compatible:st.caption('Model đã lưu nhưng không phù hợp vẫn hiện để bạn thay lựa chọn; nút tiếp tục bị khóa cho lựa chọn đó.')
    if locked:st.info('Dự án đã có vòng hoàn tất hoặc job đang chạy. Giữ model gốc; tạo dự án mới nếu cần đổi model.')
    if st.button('Xác nhận model & tiếp tục →',type='primary',disabled=locked or not all_selected):
        adaptation.save_model_selection(store,pid,settings,selected)
        st.session_state['pending_navigation']='Huấn luyện nguồn';st.rerun()


def prepare_selected(store,settings,project):
    """Return True when rendering the selected-hub preparation route."""
    path=store.path(project['id'])/'adaptation.json'
    config=read(path) if path.exists() else {};selection=config.get('model_selection',{})
    if not any(v['kind']=='hub' for v in selection.values()):return False
    registry={m['id']:m for m in load_catalog()['models']}
    from model_library import download_progress
    inventory=read(store.path(project['id'])/'dataset.json')
    profile=dataset_profile([f for f in inventory['frames'] if f['split']=='adaptation'])
    wanted=config.get('adaptation_classes',project['schema']['classes']);compatible=True
    setup=read(store.path(project['id'])/'setup.json')
    st.caption('Đây là hai model đã chọn ở bước trước. Không chọn lại, không tự tải khi mở trang.')
    for col,key in zip(st.columns(2),['A','B']):
        with col,st.container(border=True):
            value=selection.get(key,{});st.subheader('Model '+key)
            if value.get('kind')=='hub':
                m=registry.get(value['id'])
                if not m:
                    compatible=False;st.error('Model không còn trong danh mục; quay lại bước Class & model.');continue
                st.write(m.get('family',m['architecture'])+' · '+m['dataset']);chips(m['classes'])
                report=assess(m,wanted,profile,setup['source_counts'])
                if not report['eligible']:
                    compatible=False
                    st.warning('Chưa phù hợp: '+' · '.join(report['reasons']))
                    research=st.checkbox('Chỉ tải để nghiên cứu, không dùng chạy adaptation',key='prepare_research_'+project['id']+'_'+key)
                    if research:download_progress(settings,m,namespace=key,selected=True)
                else:download_progress(settings,m,namespace=key,selected=True)
            elif value.get('kind')=='source':st.info('Model này cần học nguồn. Luồng source training native nằm bên dưới.')
            else:st.success('Dùng checkpoint local đã chọn. Không tải hoặc học nguồn lại.')
    if not compatible:st.info('Quay lại Class & model để chọn model phù hợp. Tải thêm hoặc tạo binding không giải quyết các lý do trên.')
    if st.button('Cập nhật model đã chuẩn bị & kiểm tra bước tiếp →',type='primary',disabled=not compatible):
        adaptation.save_model_selection(store,project['id'],settings,selection)
        st.session_state['pending_navigation']='Vòng adaptation';st.rerun()
    if st.button('← Sửa lựa chọn model'):
        st.session_state['pending_navigation']='Class & nguồn';st.rerun()
    # Mixed source/hub selection still exposes explicitly requested native source training.
    return not any(v['kind']=='source' for v in selection.values())
