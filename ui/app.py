"""AnnoLoop local UI. Reruns never launch work without a submitted action."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ui'))
# Streamlit watches PYTHONPATH, not additions to sys.path made by the app.
# Include the backend even when users launch `streamlit run` directly.
watch_paths = [str(ROOT / 'src'), *filter(None, os.environ.get('PYTHONPATH', '').split(os.pathsep))]
os.environ['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(watch_paths))
import streamlit as st
from annoloop import jobs
from annoloop.device import probe
from annoloop.io import read
from annoloop.projects import ProjectStore
from annoloop.settings import Settings
from components import box_plot, cards, chips, empty, hero, tail
from onboarding import wizard
from inspection_page import model_inspection
from adaptation_page import page as adaptation_ui
from prediction_page import saved_predictions
from round_status import selection_status
from memory_page import memory_view
from annoloop.selection import SelectionPolicy
from annoloop.inspection import checkpoint_choices
from annoloop.setup import launch_required, refresh_training_result

st.set_page_config(page_title='AnnoLoop · Local annotation studio', page_icon='◈', layout='wide')
st.markdown('<style>'+ (ROOT/'ui/assets/style.css').read_text() + '</style>', unsafe_allow_html=True)
settings = Settings.load()
store = ProjectStore(settings.data)
PAGES = ['Tổng quan', 'Dữ liệu', 'Class & nguồn', 'Huấn luyện nguồn', 'Vòng adaptation', 'So sánh phiên bản', 'Xuất CVAT', 'Hướng dẫn & thiết bị', 'Kiểm tra model']


def navigate(name):
    st.session_state['navigation'] = name


def notice_error(error):
    st.error(str(error))
    st.caption('Kiểm tra đường dẫn và định dạng theo hướng dẫn ở cuối trang. Dữ liệu đã lưu được giữ nguyên.')


def current_project():
    key = st.session_state.get('project_id')
    if not key:
        empty('Bắt đầu bằng dataset của bạn', 'Thêm một dataset ở mục Dữ liệu để tạo dự án, đọc class và thiết lập các vòng chạy.')
        st.button('Thêm dataset →', on_click=navigate, args=('Dữ liệu',), type='primary')
        return None
    return store.get(key)


def overview():
    hero('LOCAL ANNOTATION STUDIO', 'Từ dữ liệu thô đến nhãn có thể kiểm tra.', 'Một nơi để chuẩn bị class, huấn luyện hai detector, theo dõi từng vòng và chọn phiên bản nhãn để bàn giao.')
    cards([('01 / CHỌN DỮ LIỆU','Source & target','Chọn hai folder và nhãn cần gán. Schema và inventory được chuẩn bị tự động.'),('02 / CHỌN MODEL','Pretrained trước','Dùng lại model có đủ class. Chỉ học nguồn cho detector còn thiếu nhãn.'),('03 / BÀN GIAO','Chọn nhãn & CVAT','So sánh các phiên bản, giữ lại vòng phù hợp và xuất shape hoặc track riêng.')])
    st.write('')
    p = current_project()
    if p:
        st.subheader(p['name'])
        columns = st.columns(4)
        for col,label,value in zip(columns,['Loại dữ liệu','Class trong dự án','Vòng hoàn tất','Nhãn đang chọn'],[p['schema']['data_type'].upper(),len(p['schema']['classes']),f"{len(p['completed_rounds'])}/{p['max_rounds']}",p['selected_version'] or 'Chưa chọn']):
            col.metric(label,value)
        st.button('Tiếp tục dự án →', on_click=navigate, args=('Huấn luyện nguồn',), type='primary')
    st.info('Adaptation: chọn checkpoint A/B, xác nhận trục/đơn vị, kiểm tra adapter GPU rồi gửi vòng chạy. Final predict và bàn giao CVAT còn cần hoàn thiện.')


def dataset_page():
    wizard(store, settings)


def setup_for_project(p):
    path=store.path(p['id'])/'setup.json'
    if path.is_file():return refresh_training_result(store,p['id'])
    empty('Thiết lập nhanh cho dự án', 'Chọn source, target và nhãn. Tool sẽ tự chuẩn bị các file cần thiết.')
    if st.button('Thiết lập nhanh →',type='primary'):
        st.session_state['quick_target']=read(store.path(p['id'])/'dataset.json')['root']
        st.session_state['pending_navigation']='Dữ liệu'
        st.rerun()
    return None


def class_page():
    hero('CLASS & MODEL','Class nào đã được model hỗ trợ?', 'Các số đếm được đọc từ nhãn source. Nhãn target do bạn chọn; tool không dùng gold target để kiểm tra class.')
    p=current_project()
    if not p:return
    setup=setup_for_project(p)
    if not setup:return
    chips(p['schema']['classes'])
    rows=[]
    for name in p['schema']['classes']:
        row={'Class':name,'Source train':setup['source_counts']['train'].get(name,0),'Source validation':setup['source_counts']['validation'].get(name,0)}
        for key in ('A','B'):
            row['Model '+key]='Cần học' if name in setup['plan']['models'][key]['missing_classes'] else 'Đã hỗ trợ'
        rows.append(row)
    st.dataframe(rows,width='stretch',hide_index=True)
    st.caption('Vehicle không tự tách thành car/truck/bus. Khác tên hoặc khác định nghĩa chưa có mapping được xác minh sẽ được báo thiếu class.')
    st.button('Xem kế hoạch model →',on_click=navigate,args=('Huấn luyện nguồn',),type='primary')


def source_page():
    hero('CHUẨN BỊ MODEL','Dùng pretrained trước. Chỉ học khi thiếu class.', 'Tool đã tạo inventory và YAML từ folder bạn chọn. Bạn không cần nhập các file cấu hình.')
    p=current_project()
    if not p:return
    setup=setup_for_project(p)
    if not setup:return
    plan=setup['plan'];required=[key for key,item in plan['models'].items() if item['pretrain_required']]
    bindings_path=store.path(p['id'])/'adaptation.json'
    bindings=read(bindings_path).get('models',{}) if bindings_path.exists() else {}
    for column,key in zip(st.columns(2),('A','B')):
        with column:
            with st.container(border=True):
                item=plan['models'][key]
                st.subheader(key+' · '+('SECOND-IoU' if key=='A' else 'PV-RCNN'))
                if key in bindings and item['pretrain_required']:
                    st.info('Đã chọn checkpoint trung gian cho adaptation')
                    st.caption('Run nguồn chưa hoàn tất. Có thể dùng checkpoint này để thử mà không bắt buộc train lại model này.')
                elif item['pretrain_required']:
                    st.info('Cần học: '+', '.join(item['missing_classes']))
                    st.caption('Head sẽ học class đã chọn, giữ lại weights phần còn tương thích. Các class cũ được chọn cũng có trong nhãn train để tránh quên.')
                else:
                    st.success('Dùng checkpoint hiện có · bỏ qua pretrain')
    blockers=plan['source_blockers']+list(plan['config_blockers'].values())
    for reason in dict.fromkeys(blockers):st.warning(reason)
    selected_models=required
    if required:
        selected_models=st.multiselect('Model cần học nguồn trong lượt này',required,
            default=[key for key in required if key not in bindings],key='source_detectors_'+p['id'])
        st.caption('Có thể chỉ chọn B nếu muốn dùng checkpoint A trung gian đã lưu. Lựa chọn này không tự gửi job.')
        if set(selected_models)=={'A','B'}:
            st.info('Chạy A → B: dừng sớm A sẽ chuyển sang B sau khi A dừng hẳn và GPU rảnh. A lỗi sẽ dừng chuỗi.')
    if not required:
        st.success('Cả hai detector đã hỗ trợ đủ nhãn bạn chọn. Không cần train nguồn lại.')
        st.button('Tiếp tục adaptation →',on_click=navigate,args=('Vòng adaptation',),type='primary',disabled=bool(blockers))
    elif setup['launches']:
        receipt=setup['launches'][-1]
        if receipt['status'] in ('failed','dispatch_failed'):
            st.warning('Lần chạy trước đã lỗi và đã dừng. Chưa có lượt thử lại mới; thông báo và log bên dưới thuộc lần chạy đã lưu.')
            with st.expander('Xem lỗi của lần chạy trước'):
                if receipt.get('error'):st.error(receipt['error'])
                if receipt.get('path'):st.caption(receipt['path'])
            st.caption('Không cần tạo lại dự án. Thử lại sẽ tạo run mới, kiểm tra môi trường worker và giữ nguyên mọi log/checkpoint cũ.')
            if st.button('Thử lại với môi trường đã sửa',type='primary',disabled=bool(blockers) or not selected_models):
                with st.spinner('Kiểm tra runtime model trước khi tạo run mới…'):
                    launch_required(store,p['id'],settings,detectors=selected_models)
                st.rerun()
        else:
            status_label={'completed_with_stops':'Đã xử lý chuỗi, có model dừng sớm',
                          'cancelled':'Đã hủy chuỗi'}.get(receipt['status'],receipt['status'])
            st.info('Job đã gửi: '+status_label+'. Đổi trang hoặc refresh không gửi lại.')
            if receipt.get('error'):st.error(receipt['error'])
            if receipt.get('path'):st.caption(receipt['path'])
            if receipt.get('stopped_detectors'):
                st.warning('Model dừng sớm: '+', '.join(receipt['stopped_detectors'])+'. Checkpoint trung gian được giữ; model hoàn tất còn lại vẫn được nhận.')
            if receipt['status'] in ('completed','completed_with_stops','cancelled') and required:
                if st.button('Học các class còn thiếu',type='primary',disabled=bool(blockers) or not selected_models):
                    launch_required(store,p['id'],settings,detectors=selected_models)
                    st.rerun()
    else:
        st.caption('Source training chỉ chạy cho model thiếu class. A và B chạy tuần tự, lưu checkpoint mỗi 10 epoch + cuối. Runtime được kiểm tra trước khi gửi job; log và checkpoint được giữ nếu bạn dừng.')
        if st.button('Học các class còn thiếu',type='primary',disabled=bool(blockers) or not selected_models):
            with st.spinner('Kiểm tra runtime model…'):
                result=launch_required(store,p['id'],settings,detectors=selected_models)
            st.success('Đã gửi job: '+result.get('path',''))
            st.rerun()
    with st.expander('Chi tiết kỹ thuật · tool đã tự tạo'):
        st.write('Source: '+setup['loader_root']);st.write('Target: '+setup['target_root'])
        st.caption('File được lưu trong project/generated; không ghi vào folder dataset gốc.')
        st.json(plan)
    if checkpoint_choices(store,p['id'],settings):
        st.button('Kiểm tra checkpoint và xem dự đoán →',on_click=navigate,args=('Kiểm tra model',),type='primary')
    monitor()


def inspection_page():
    p=current_project()
    if p:model_inspection(store,settings,p)


@st.fragment(run_every=5)
def monitor():
    st.subheader('Theo dõi huấn luyện')
    paths=sorted((settings.data/'jobs').glob('*/job.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not paths:
        empty('Chưa có job huấn luyện', 'Khi bắt đầu, trạng thái tiến trình, epoch, loss và checkpoint sẽ xuất hiện ở đây. Tự cập nhật mỗi 5 giây.')
        return
    labels={str(p.parent): (read(p).get('model') or jobs.model_identity(read(p).get('spec', {})))['label']
            +' · '+p.parent.name[:8] for p in paths}
    selected=st.selectbox('Run đã đăng ký',list(labels),format_func=lambda p:labels[p])
    snapshot=jobs.status(selected); log=snapshot['log']; receipt=snapshot['training']
    st.subheader('Model: '+snapshot['model']['label'])
    a,b,c=st.columns(3)
    a.metric('Tiến trình','Đang sống' if snapshot['process_alive'] else 'Đã dừng / chưa khởi chạy')
    b.metric('Trạng thái receipt',receipt.get('status',snapshot['job']['status']))
    c.metric('Loss',str(log.get('loss',{}).get('value',log.get('loss',{}).get('source','—'))))
    if log.get('epochs'):
        st.progress(min(1.,log['epoch_completed']/log['epochs']),text=f"Epoch {log['epoch_completed']} / {log['epochs']}")
    else: st.caption('Chưa có epoch trong log; không suy ra tiến độ từ thời gian đã chạy.')
    st.code(tail(Path(selected)/'train.log'),language='log',height=250)
    files=list(Path(selected).rglob('*.pth'))
    st.caption(f'{len(files)} checkpoint trên đĩa · lưu mỗi 10 epoch + cuối')
    with st.expander('Checkpoint & thông tin run'):
        st.write([str(p) for p in files]);st.json(receipt)
    pair=jobs.pair_for_run(selected,settings)
    help_text='Giữ log/checkpoint. Nếu đây là A của chuỗi nguồn A → B, dừng A sẽ tự chuyển sang B.'
    if st.button('Dừng trainer này',disabled=not snapshot['process_alive'],help=help_text):
        if jobs.stop(selected): st.success('Đã gửi tín hiệu dừng. Chờ cập nhật PID.')
        else: st.info('Tiến trình đã kết thúc.')
    if pair:
        if pair['detector']=='A' and pair['pair']['status'] in ('starting','running'):
            st.caption('Dừng riêng A sẽ chuyển sang B. Muốn hủy toàn bộ lượt chạy, chọn nút bên dưới.')
        if st.button('Dừng cả chuỗi A → B',disabled=pair['pair']['status'] not in ('starting','running'),
                     help='Ngắt model hiện tại và hủy model còn chờ; giữ toàn bộ artifacts.'):
            jobs.stop_source_pair(pair['path'])
            st.success('Đã yêu cầu dừng toàn bộ chuỗi.')


def rounds_page():
    p=current_project()
    if p:adaptation_ui(store,settings,p)


def compare_page():
    hero('05 / SO SÁNH','Giữ vòng tốt, quay lại khi cần.', 'So sánh nhãn trên cùng frame sample. Vòng mới nhất và phiên bản đang chọn được hiển thị riêng để tránh dùng nhầm nhãn.')
    p=current_project()
    if not p:return
    saved_predictions(store,p)
    versions=p['completed_rounds']
    if not versions:
        empty('Chưa có snapshot nhãn chung','Dự đoán đã lưu của từng detector hiển thị phía trên nếu có. Nhãn hợp nhất và lựa chọn rollback xuất hiện khi vòng hoàn tất.')
        return
    st.success('Phiên bản đang dùng: '+str(p['selected_version'] or 'Chưa chọn'))
    selection_status(store,p)
    memory_view(store,p)
    root=store.path(p['id'])/'rounds'
    metrics=[{'Vòng':v,**read(root/v/'round.json')['metrics']} for v in versions]
    st.dataframe(metrics,width='stretch',hide_index=True)
    st.caption('Agreement và point support là proxy trên target chưa có nhãn. Không diễn giải thành accuracy, precision hay recall.')
    a,b=st.columns(2)
    before=a.selectbox('Trước',versions,index=max(0,len(versions)-2))
    after=b.selectbox('Sau',versions,index=len(versions)-1)
    frame=st.selectbox('Frame sample cố định',p['sample_frame_ids'])
    documents=[read(root/v/'samples.json')[frame] for v in (before,after)]
    figures=[box_plot(doc,p['schema']['data_type']) for doc in documents]
    # Shared bounds make geometric movement directly comparable across rounds.
    xs=[x for fig in figures for trace in fig.data for x in trace.x]
    ys=[y for fig in figures for trace in fig.data for y in trace.y]
    if xs and ys:
        for fig in figures:
            fig.update_xaxes(range=[min(xs)-2,max(xs)+2]);fig.update_yaxes(range=([max(ys)+2,min(ys)-2] if p['schema']['data_type']=='2d' else [min(ys)-2,max(ys)+2]),autorange=False)
    for index,(col,version,document,figure) in enumerate(zip(st.columns(2),(before,after),documents,figures)):
        with col:
            st.subheader(version+f" · {len(document['boxes'])} box")
            st.plotly_chart(figure,width='stretch',key='chart_'+str(index))
            st.caption('Hình học box nhìn từ trên xuống (BEV); chưa overlay point cloud.' if p['schema']['data_type']=='3d' else 'Hình học bbox theo tọa độ pixel; chưa overlay ảnh.')
            with st.expander('Chi tiết class / confidence / track'):
                st.json(document)
    with st.form('select_round'):
        chosen=st.selectbox('Phiên bản nhãn muốn sử dụng',versions,index=versions.index(p['selected_version']) if p['selected_version'] in versions else len(versions)-1)
        accept=st.form_submit_button('Dùng phiên bản này',type='primary')
    if accept:
        store.select(p['id'],chosen);st.success('Đã chọn '+chosen+'. Các vòng còn lại được giữ nguyên.');st.rerun()
    with st.expander('Dự đoán riêng của hai detector'):
        round_info=read(root/after/'round.json')
        work_directory=Path(round_info['work_directory']) if round_info.get('work_directory') else store.path(p['id'])/'work'/after
        artifact=work_directory/'model_samples.json'
        if artifact.exists():st.json(read(artifact))
        else:st.info('Vòng này chưa có artifact sample riêng A/B. Đồ thị trên là nhãn hợp nhất.')


def cvat_page():
    hero('06 / CVAT','Bàn giao đúng phiên bản nhãn.', 'Chọn shape cho box độc lập hoặc track cho đối tượng xuyên frame. Hai chế độ tạo task riêng, giữ rõ nguồn nhãn và frame mapping.')
    p=current_project()
    if not p:return
    st.metric('Phiên bản nhãn được chọn',p['selected_version'] or 'Chưa chọn')
    mode=st.radio('Kiểu annotation',['Shape · box theo frame','Track · đối tượng theo chuỗi'],horizontal=True)
    st.caption('Track cần timestamp và pose hoặc xác nhận sensor cố định. Nội suy chưa đủ bằng chứng chỉ là ignore / review.')
    st.text_input('Địa chỉ CVAT local',value='http://localhost:8080')
    st.warning('Chưa sẵn sàng bàn giao: final inference trên toàn bộ dataset, chuẩn hóa media và kết nối UI → exporter còn cần hoàn thiện. Snapshot adaptation chưa phải kết quả final cho toàn bộ dữ liệu.')
    st.button('Tạo task CVAT',disabled=True,type='primary')
    if p['selected_version']:
        version=p['selected_version'];path=store.path(p['id'])/'rounds'/version/'labels.json'
        st.download_button('Tải snapshot adaptation (.json)',path.read_bytes(),file_name=version+'_adaptation_labels.json',mime='application/json')
        st.caption('File snapshot nội bộ để kiểm tra; không phải gói import CVAT.')
    cards([('01','Chọn phiên bản','Dùng vòng đã kiểm tra; rollback ở màn So sánh.'),('02','Final predict','Chạy trên toàn bộ tập bàn giao, giữ thứ tự frame.'),('03','Task riêng','Xuất shape hoặc track và kiểm tra lại nhãn từ CVAT.')])


def help_page():
    hero('HƯỚNG DẪN','Từng bước, từ dữ liệu của bạn.', 'Các thao tác tạo job chỉ chạy khi bạn bấm bắt đầu. Đổi trang hoặc refresh không khởi động lại trainer.')
    a,b=st.columns([2,1])
    with a:
        with st.expander('Hướng dẫn đầy đủ',expanded=True):
            st.markdown((ROOT/'docs/USER_GUIDE.md').read_text())
        st.download_button('Tải hướng dẫn (.md)',(ROOT/'docs/USER_GUIDE.md').read_text(),file_name='ANNOLOOP_USER_GUIDE.md')
    with b:
        with st.container(border=True):
            st.subheader('Thiết bị & môi trường')
            st.caption('NVIDIA GPU · một worker tại một thời điểm')
            if st.button('Kiểm tra CUDA',type='primary'):
                with st.spinner('Kiểm tra venv và CUDA…'):st.session_state['hardware']=probe(settings.trainer_python)
            if 'hardware' in st.session_state:
                value=st.session_state['hardware']
                (st.success if value.get('ready') else st.error)('CUDA sẵn sàng' if value.get('ready') else 'CUDA chưa sẵn sàng')
                st.json(value)
                st.caption('CUDA available chưa xác nhận compiled ops ST3D tương thích.')
            st.caption('Python venv');st.code(str(settings.python),language=None)
            st.caption('Python worker GPU');st.code(str(settings.trainer_python),language=None)
            st.caption('Nơi lưu dự án');st.code(str(settings.data),language=None)
            st.caption('Đổi bằng VINAI_PYTHON / ANNOLOOP_DATA_DIR trước khi mở app. Chưa có Docker trong bản này.')


if next_page := st.session_state.pop('pending_navigation', None):
    st.session_state['navigation'] = next_page

if pending := st.session_state.pop('pending_project_id', None):
    st.session_state['project_id'] = pending

st.session_state.setdefault('navigation', 'Dữ liệu')

with st.sidebar:
    st.markdown('<div class="brand"><span>◈</span>AnnoLoop</div><div class="muted">Local annotation studio</div>',unsafe_allow_html=True)
    st.write('')
    projects=sorted(store.list(),key=lambda p:p['created_at'],reverse=True)
    ids=[p['id'] for p in projects]
    if st.session_state.get('project_id') not in ids:
        st.session_state['project_id']=ids[0] if ids else None
    labels={p['id']:p['name']+' · '+p['id'][:6] for p in projects}
    st.selectbox('Dự án hiện tại',ids if ids else [None],key='project_id',
                 format_func=lambda p:labels.get(p,'Chưa có dự án'))
    if st.session_state['project_id']:
        selected_project=store.get(st.session_state['project_id'])
        inventory=read(store.path(selected_project['id'])/'dataset.json')
        with st.container(border=True):
            chips([selected_project['schema']['data_type'].upper(),f"{len(inventory['frames'])} frame"])
            st.caption('Nhãn: '+', '.join(selected_project['schema']['classes']))
            st.caption(f"Vòng {len(selected_project['completed_rounds'])}/{selected_project['max_rounds']} · Nhãn đang dùng: {selected_project['selected_version'] or 'chưa chọn'}")
            with st.expander('Thiết lập dự án'):
                with st.form('project_settings_'+selected_project['id']):
                    name=st.text_input('Tên dự án',value=selected_project['name'])
                    rounds=st.number_input('Số vòng tối đa',min_value=len(selected_project['completed_rounds']),max_value=200,value=selected_project['max_rounds'])
                    mode=st.selectbox('Cách chạy huấn luyện',['manual','automatic'],
                        index=['manual','automatic'].index(selected_project['execution_mode']),format_func=lambda x:'Từng vòng' if x=='manual' else 'Tự động')
                    policy=selected_project['selection_policy']
                    selection=st.selectbox('Cách chọn phiên bản nhãn',['manual','automatic'],
                        index=['manual','automatic'].index(policy['mode']),format_func=lambda x:'Tôi xem và chọn' if x=='manual' else 'Theo ngưỡng')
                    agreement=st.slider('Agreement tối thiểu',0.,1.,policy['min_agreement'],.05)
                    support=st.slider('Point support tối thiểu',0.,1.,policy['min_supported_fraction'],.05)
                    st.caption('Ngưỡng dùng proxy không gold; không chứng minh accuracy.')
                    submitted=st.form_submit_button('Lưu dự án')
                if submitted:
                    try:
                        store.configure(selected_project['id'],name,int(rounds),mode,
                            SelectionPolicy(**dict(policy,mode=selection,min_agreement=agreement,min_supported_fraction=support)))
                        st.rerun()
                    except ValueError as error:st.error(str(error))
    st.button('＋ Tạo dự án mới',on_click=navigate,args=('Dữ liệu',),width='stretch')
    st.radio('Không gian làm việc',PAGES,key='navigation',label_visibility='collapsed')
    st.divider()
    st.caption('LOCAL WORKSPACE · MVP')
    st.caption('Source train có nhãn. Target adaptation không dùng gold. Mỗi phiên bản được lưu riêng.')

page=st.session_state['navigation']
st.caption('ANNOLOOP  /  '+page.upper())
if message:=st.session_state.pop('flash',None):st.success(message)
try:
    dict(zip(PAGES,[overview,dataset_page,class_page,source_page,rounds_page,compare_page,cvat_page,help_page,inspection_page]))[page]()
except Exception as error:
    notice_error(error)
