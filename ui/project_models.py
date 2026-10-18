"""The single project model-selection step; all later pages consume this binding."""
from pathlib import Path
import streamlit as st
from annoloop import adaptation
from annoloop.io import read
from annoloop.model_hub import load_catalog,canonical,model_root
from annoloop.model_filter import dataset_profile,assess
from annoloop.native_sources import catalog as native_catalog
from annoloop.model_names import friendly,verdict
from components import chips

BADGES={'usable':'✅ Dùng được','download':'⬇ Cần tải','unusable':'⛔ Chưa dùng được'}
RANK={'usable':0,'download':1,'unusable':2}


def project_profile(store,pid):
    inventory=read(store.path(pid)/'dataset.json')
    return dataset_profile([f for f in inventory['frames'] if f['split']=='adaptation'])


def selection_step(store,settings,project,setup):
    pid=project['id'];path=store.path(pid)/'adaptation.json'
    config=read(path) if path.exists() else {};saved=config.get('model_selection',{})
    choices=adaptation.model_choices(store,pid,settings)
    wanted=config.get('adaptation_classes',project['schema']['classes'])
    locked=bool(project['completed_rounds'] or adaptation.execution_busy(store,pid))
    st.subheader('Hai model cho toàn bộ workflow')
    st.caption('Chọn một lần tại đây. Bước Chuẩn bị model tải/học khi cần; Adaptation dùng đúng lựa chọn đã lưu.')
    profile=project_profile(store,pid)
    with st.container(border=True):
        st.markdown('**Kiểm tra đầu vào trước khi chọn model**')
        chips(profile['features'] or ['Chưa xác định feature'])
        st.caption(f"Đã đọc header {profile['checked']}/{profile['total']} frame mẫu; không chạy detector, không đọc nhãn target. RGB không thay cho intensity.")
        if profile['mixed']:st.warning('Các frame mẫu có feature khác nhau. Bộ lọc dùng những feature có ở tất cả mẫu đọc được.')
        if profile['errors']:
            st.warning('Chưa xác định đủ feature đầu vào. Không tự coi model là tương thích.')
            with st.expander('Chi tiết frame chưa đọc được'):st.dataframe(profile['errors'],hide_index=True,width='stretch')
        st.caption('Dataset hiện chưa có manifest multi-sweep/timestamp/ego pose được adapter hỗ trợ.')
    modes=['strict','current_scan_only']
    input_mode=st.radio('Cách dùng đầu vào MMDetection3D',modes,index=modes.index(config.get('mmdet_input_mode','strict')),
        format_func=lambda v:'Giữ yêu cầu đầu vào gốc' if v=='strict' else 'Mỗi file là scan hiện tại · Δt=0, không lịch sử',
        key='workflow_input_mode_'+pid,disabled=locked,
        help='Chọn chế độ thứ hai chỉ khi mỗi file là một scan riêng. Không tạo sweep/intensity, không đổi weights. Chất lượng có thể giảm so với multi-sweep gốc; vẫn cần kiểm chứng GPU và calibration.')
    compatible=st.checkbox('Chỉ hiện model dùng được hoặc cần tải',value=True,key='workflow_compatible_'+pid,
        help='Ẩn model đã biết là chưa chạy được (head/framework chưa có recipe, checkpoint v0.x, sai đầu vào). Model cần tải vẫn hiện và được kiểm tra lại sau tải.')
    if not compatible:st.warning('Chế độ xem mở rộng: có thể chọn để chuẩn bị model, nhưng model không tương thích vẫn không chạy được adaptation.')
    full=st.checkbox('Chỉ hiện checkpoint đủ class',value=True,key='workflow_complete_'+pid)
    registry=load_catalog()['models'];selected={};all_selected=True;excluded={}
    # One directory listing instead of resolving 62 cache paths on every rerun.
    cache_dir=Path(settings.data)/'model_cache'
    present={p.name for p in cache_dir.iterdir()} if cache_dir.is_dir() else set()
    native_registry=native_catalog()
    with st.expander('Nguồn ST3D / ST3D++ và các checkpoint liên quan'):
        st.link_button('ST3D chính thức · Model Zoo','https://github.com/CVMI-Lab/ST3D#model-zoo')
        st.caption('ST3D có SECOND-IoU, PV-RCNN, PointRCNN. Link repo không tự chứng nhận checkpoint/class hoặc khả năng chạy. PointRCNN chưa nối đủ trong MVP.')
        st.caption('Hai lựa chọn 3DOD-LSF bên dưới có payload SHA256 đã ghi nhận, class gốc Vehicle và đầu vào XYZ. Link archive sẽ được kiểm tra khi bạn bấm tải; không tải lúc xem trang.')
    for col,key in zip(st.columns(2),['A','B']):
        with col,st.container(border=True):
            options={};labels={};metadata={};reports={};ranks={}
            current=saved.get(key)
            if current is None and key in config.get('models',{}):
                m=config['models'][key];current=dict(kind='hub',id=m['model_id']) if m.get('backend')=='registry_framework' else dict(kind='local',path=m['path'])
            for m in choices[key]:
                if m.get('backend','native_st3d')!='native_st3d':continue
                report=assess(dict(m,backend='native_st3d'),wanted,profile,setup['source_counts'])
                token='local:'+m['path'];value=dict(kind='local',path=m['path'])
                if not report['eligible']:excluded[token]={'Model':m['name'],'Lý do chưa phù hợp':' · '.join(report['reasons'])}
                if compatible and not report['selectable'] and current!=value:continue
                options[token]=value;labels[token]=m['name']+' · local';metadata[token]=m;reports[token]=report
            for m in registry:
                from annoloop.mmdet_recipe import cached_model
                m=cached_model(settings.data,m) if m['id'] in present else m
                if m.get('mmdet_recipe'):
                    from annoloop.mmdet_recipe import effective
                    m=dict(m,mmdet_recipe=effective(m['mmdet_recipe'],input_mode))
                report=assess(m,wanted,profile,setup['source_counts']);enough=not report['missing_classes']
                cached=m['id'] in present and (model_root(settings.data,m['id'])/'installed.json').exists()
                state,why=verdict(m,report,cached);title=friendly(m)
                if state!='usable':excluded[m['id']]={'Model':title,'Trạng thái':BADGES[state],'Lý do chưa phù hợp':why}
                is_current=current==dict(kind='hub',id=m['id'])
                # Default list: models that can run now or after download; the rest stay in the table.
                hidden=m.get('modality')!='point_cloud' or compatible and state=='unusable' or full and not enough
                if hidden and not is_current:continue
                token='hub:'+m['id'];options[token]=dict(kind='hub',id=m['id']);ranks[token]=RANK[state]
                labels[token]=BADGES[state]+' · '+title+' · '+m['framework']
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
            if compatible and not any(r['selectable'] for r in reports.values()):
                st.info('Chưa có checkpoint qua bộ lọc cho vai trò này. Xem lý do bên dưới; tool không tự train hoặc đổi model.')
            values=[None]+sorted(options,key=lambda v:ranks.get(v,0));initial=next((v for v in options if options[v]==current),None)
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
                if report['input_status']=='compatible':st.caption('Khớp class/đầu vào. Adapter và khả năng adaptation được kiểm tra ở bước Chuẩn bị model.')
                elif report['input_status']=='pending':st.info('Đủ class; input chưa có bằng chứng đầy đủ. Có thể chọn để tải và chuẩn bị, chưa cấp quyền chạy adaptation.')
                else:
                    st.warning('Chưa phù hợp: '+' · '.join(report['reasons']))
                    if compatible and not report['selectable']:all_selected=False
                missing=[c for c in wanted if canonical(c) not in {canonical(n) for n in m['classes']}]
                (st.warning if missing else st.success)('Thiếu: '+', '.join(missing) if missing else 'Có đủ class yêu cầu')
                if pick.startswith('hub:'):
                    st.caption('Nguồn: '+m['framework'])
                    st.caption('Sau tải: script nối recipe framework, kiểm tra predict/backward/ignore/calibration rồi báo đạt hoặc phần còn thiếu.')
    if excluded:
        with st.expander(f"{len(excluded)} model cần tải hoặc chưa dùng được · xem lý do"):
            st.dataframe(list(excluded.values()),width='stretch',hide_index=True)
            st.caption('Thiếu feature đã biết sẽ chặn lựa chọn. Thiếu adapter không ẩn checkpoint; script sau tải sẽ báo bước chưa triển khai. Không tự train class mới để che lỗi adapter.')
    if compatible:st.caption('✅ đã tải và nối recipe · ⬇ cần tải, loại head đã có recipe. Cả hai vẫn phải qua kiểm tra GPU ở bước 3.')
    if locked:st.info('Dự án đã có vòng hoàn tất hoặc job đang chạy. Giữ model gốc; tạo dự án mới nếu cần đổi model.')
    unchanged=all_selected and bool(saved) and saved==selected and config.get('mmdet_input_mode','strict')==input_mode
    if unchanged:
        # Nothing to rewrite: move on without re-saving the binding.
        st.button('Tiếp tục: chuẩn bị model →',type='primary',key='models_continue_'+pid,
                  on_click=lambda:st.session_state.update(navigation='Huấn luyện nguồn'))
        return
    if not locked and not all_selected:
        missing=[k for k in ('A','B') if k not in selected]
        st.caption('Chọn model '+' và '.join(missing)+' để tiếp tục.' if missing else
                   'Model đang chọn chưa khớp class/đầu vào; đổi model hoặc tắt bộ lọc để chỉ tải nghiên cứu.')
    if st.button('Xác nhận model & tiếp tục →',type='primary',disabled=locked or not all_selected):
        adaptation.save_model_selection(store,pid,settings,selected,mmdet_input_mode=input_mode)
        st.session_state['pending_navigation']='Huấn luyện nguồn';st.rerun()


def prepare_selected(store,settings,project):
    """Return True when rendering the selected-hub preparation route."""
    path=store.path(project['id'])/'adaptation.json'
    config=read(path) if path.exists() else {};selection=config.get('model_selection',{})
    if not any(v['kind']=='hub' for v in selection.values()):return False
    registry={m['id']:m for m in load_catalog()['models']}
    from model_library import download_progress
    profile=project_profile(store,project['id'])
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
                # Same evidence as step 2: downloaded recipe plus the saved input mode.
                from annoloop.mmdet_recipe import cached_model,effective
                m=cached_model(settings.data,m)
                if m.get('mmdet_recipe'):m=dict(m,mmdet_recipe=effective(m['mmdet_recipe'],config.get('mmdet_input_mode','strict')))
                st.write(friendly(m));chips(m['classes'])
                report=assess(m,wanted,profile,setup['source_counts'])
                if not report['selectable']:
                    compatible=False
                    st.warning('Chưa phù hợp: '+' · '.join(report['reasons']))
                    research=st.checkbox('Chỉ tải để nghiên cứu, không dùng chạy adaptation',key='prepare_research_'+project['id']+'_'+key)
                    if research:download_progress(settings,m,namespace=key,selected=True)
                else:
                    st.caption('Nút chung bên dưới sẽ tải và kiểm tra model này.')
                    with st.expander('Nâng cao · chỉ tải / chuẩn bị model này'):
                        download_progress(settings,m,namespace=key,selected=True)
            elif value.get('kind')=='source':st.info('Model này cần học nguồn. Luồng source training native nằm bên dưới.')
            else:st.success('Dùng checkpoint local đã chọn. Không tải hoặc học nguồn lại.')
    if not compatible:st.info('Quay lại Class & model để sửa class/đầu vào không tương thích. Không tự tải model khác.')
    with st.expander('Nâng cao · cập nhật binding có sẵn'):
        st.caption('Dùng khi đã chuẩn bị bằng CLI. Nút chung bên dưới tự thực hiện bước này.')
        if st.button('Cập nhật model đã chuẩn bị & kiểm tra bước tiếp →',disabled=not compatible):
            adaptation.save_model_selection(store,project['id'],settings,selection)
            st.rerun()
    if st.button('← Sửa lựa chọn model'):
        st.session_state['pending_navigation']='Class & nguồn';st.rerun()
    from preparation_page import panel
    panel(store,settings,project)
    # Mixed source/hub selection still exposes explicitly requested native source training.
    return not any(v['kind']=='source' for v in selection.values())
