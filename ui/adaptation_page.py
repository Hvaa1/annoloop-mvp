"""Configure real native adapters and dispatch only explicit user actions."""
from pathlib import Path
from dataclasses import asdict
import streamlit as st
from annoloop import adaptation
from annoloop.io import read
from annoloop.previews import inference_progress
from annoloop.resume import options as resume_options
from annoloop.pseudo_memory import MemoryPolicy
from components import hero, tail, live_fragment
from round_status import selection_status


STAGES = {'predict_target': 'Predict target', 'calibrate_source_validation': 'Calibration source validation',
          'train_target': 'Train trên pseudo-label chung + source', 'fuse_and_snapshot': 'Hợp nhất và lưu phiên bản',
          'verify_native_gpu': 'Kiểm tra forward/backward GPU · không optimizer step',
          'load_pseudo_memory': 'Đọc lịch sử pseudo-label', 'update_pseudo_memory': 'Cập nhật bộ nhớ nhãn'}


def page(store, settings, project):
    hero('VÒNG ADAPTATION', 'Chọn model, kiểm tra rồi chạy.',
         'A và B chạy lần lượt. Mỗi vòng lưu nhãn, checkpoint và sample cố định để bạn xem hoặc quay lại.')
    pid = project['id']
    from annoloop.preparation_jobs import active as preparing
    if preparing(store,pid):
        st.info('Model đang tải/chuẩn bị/kiểm chứng. Theo dõi tại bước 3 · Chuẩn bị model; chưa chạy vòng adaptation.')
        return
    a, b, c = st.columns(3)
    a.metric('Vòng hoàn tất', f"{len(project['completed_rounds'])}/{project['max_rounds']}")
    b.metric('Huấn luyện', 'Từng vòng' if project['execution_mode'] == 'manual' else 'Tự động')
    c.metric('Nhãn đang dùng', project['selected_version'] or 'Chưa chọn')
    from annoloop.final_delivery import active as delivery_active
    if delivery_active(store,pid):
        st.info('Final predict đang chạy. Theo dõi tại Xuất CVAT; chờ lượt này dừng trước khi train tiếp.')
        if st.button('Xem final predict →'):
            st.session_state['pending_navigation']='Xuất CVAT';st.rerun()
        return
    st.caption('Chọn nhãn: ' + ('tự động theo ngưỡng' if project['selection_policy']['mode'] == 'automatic' else
                              'duyệt từng vòng bằng tay'))
    selection_status(store, project)
    st.markdown('<div class="flow"><span>Calibration nguồn A → B</span><span>Predict A → B</span><span>Nhãn chung</span>'
                '<span>Train A → B</span><span>Sample & chọn vòng</span></div>', unsafe_allow_html=True)
    check = adaptation.readiness(store, pid, fast=True)
    config = check['config']
    if check.get('implementation_missing'):
        st.info('Model đã có thể tải/bind nhưng recipe chưa đủ để train adaptation. '
                'Đây là bước thiếu code, không phải kiểm tra GPU đang chờ. '
                'Xem checklist ở Chuẩn bị model; kiểm tra lại không tự bổ sung trainer. '
                'Bạn có thể giữ các checkpoint này và sửa lựa chọn ở bước 2 nếu muốn dùng recipe native hiện có.')
        if st.button('Xem thành phần adapter còn thiếu →',key='missing_recipe_'+pid):
            st.session_state['pending_navigation']='Huấn luyện nguồn';st.rerun()
    active_path = adaptation.active_job(store, pid)
    active = bool(active_path)
    st.session_state[f'adaptation_waiting_{pid}'] = active_path
    with st.container(border=True):
        st.subheader('1. Model đã chọn cho dự án')
        selected = {key:config['models'].get(key,{}).get('path') for key in ('A','B')}
        for col, key in zip(st.columns(2), ('A', 'B')):
            with col:
                candidate=config['models'].get(key)
                st.markdown('**Model '+key+'**')
                if candidate:
                    st.write(candidate['architecture']);st.caption(candidate.get('name',Path(candidate['path']).name))
                    if candidate.get('backend')=='registry_framework':
                        from annoloop.hub_selection import blockers as hub_blockers
                        st.caption('Kho model · '+candidate['architecture']+' · '+candidate['framework'])
                        st.caption('Class checkpoint: '+', '.join(candidate['classes']))
                        for reason in hub_blockers(candidate,config.get('adaptation_classes',project['schema']['classes'])):st.warning(reason)
                    elif candidate['intermediate']:
                        st.caption('Checkpoint trung gian. Dùng để thử hướng đi; kiểm tra chất lượng trước khi dùng nhãn tự động.')
                else:
                    planned=config.get('model_selection',{}).get(key,{})
                    if planned.get('kind')=='hub':
                        from annoloop.model_hub import load_catalog
                        meta=next((m for m in load_catalog()['models'] if m['id']==planned['id']),None)
                        if meta:st.write(meta['architecture']+' · '+meta['dataset']);st.caption(meta['name'])
                        st.info('Đã chọn model; chưa hoàn tất bước Chuẩn bị model.')
                    elif planned.get('kind')=='source':st.info('Đã chọn học nguồn; chờ checkpoint ở bước Chuẩn bị model.')
                    else:st.info('Chưa chọn model. Quay lại bước 2.')
        if st.button('← Quay lại Class & model',disabled=active):
            st.session_state['pending_navigation']='Class & nguồn';st.rerun()
        st.caption('Adaptation dùng model đã lưu ở bước Class & model. Trang này chỉ xác nhận dữ liệu và cấu hình vòng chạy.')
        if project['completed_rounds']:
            st.info('Dự án đã có vòng hoàn tất: giữ checkpoint khởi tạo để bảo toàn lịch sử. Muốn đổi model gốc, tạo dự án mới từ cùng dataset.')
        scope_key = f'adapt_classes_{pid}'
        saved_scope = config.get('adaptation_classes', project['schema']['classes'])
        stamp_key = f'adapt_scope_saved_at_{pid}'
        if scope_key not in st.session_state or st.session_state.get(stamp_key) != config.get('saved_at'):
            st.session_state[scope_key] = saved_scope
        st.session_state[stamp_key] = config.get('saved_at')
        chosen_classes = st.multiselect('Class chạy adaptation', project['schema']['classes'],
            default=None, key=scope_key, disabled=active or bool(project['completed_rounds']))
        st.caption('Chỉ các class được chọn cần calibration và được tạo nhãn dương target. '
                   'Box dự đoán thuộc class còn lại được giữ ở vùng ignore. '
                   'Giữ nguyên đầu ra checkpoint và source loss; không train pretrain lại khi đổi lựa chọn trước vòng đầu.')
        st.subheader('2. Xác nhận dữ liệu target')
        confirmed = st.checkbox('Target dùng mét; X hướng trước, Y trái, Z lên trong hệ tọa độ LiDAR',
                                value=config.get('coordinates_confirmed', False), key=f'adapt_axes_{pid}', disabled=active)
        columns = st.columns(3)
        bin_columns = columns[0].selectbox('BIN · số cột mỗi điểm', [4, 3, 5],
                    index=[4, 3, 5].index(config.get('bin_columns', 4)), key=f'adapt_bin_{pid}', disabled=active,
                    help='KITTI Velodyne là 4 cột x, y, z, intensity. BIN có khai báo format sẽ được đối chiếu trước kiểm tra GPU. PCD đọc header, không dùng số cột BIN.')
        epochs = columns[1].number_input('Epoch mỗi model / vòng', min_value=1, max_value=200,
                    value=config.get('epochs', 1), key=f'adapt_epochs_{pid}', disabled=active)
        learning_rate = columns[2].number_input('Learning rate adaptation', min_value=.000001,
                    max_value=.01, value=config.get('learning_rate', .0001), format='%.6f',
                    key=f'adapt_lr_{pid}', disabled=active)
        with st.expander('Source loss & tracking'):
            source_weight = st.slider('Trọng số source loss', .1, 1., config.get('source_weight', .5), .1,
                                      key=f'adapt_source_weight_{pid}', disabled=active)
            st.caption('Source có nhãn giữ kiến thức cũ; target dùng pseudo-label. Adam với LR đã chọn. '
                       'Vùng bất đồng được ignore trong dense/ROI/point loss. Frame không có nhãn dương đủ điểm được bỏ qua khi train.')
            st.info('Mặc định không tracking. Tracking để chọn nhãn train cần sequence, timestamp và ego pose hợp lệ; không giả định cảm biến đứng yên.')
        saved_memory = asdict(MemoryPolicy(**config.get('memory_policy', {})))
        memory_policy = dict(saved_memory)
        with st.expander('Bộ nhớ pseudo-label qua các vòng'):
            memory_policy['enabled'] = st.checkbox('Giữ lịch sử từng box', value=saved_memory['enabled'],
                key=f'adapt_memory_enabled_{pid}', disabled=active)
            left, right = st.columns(2)
            memory_policy['stable_hits'] = int(left.number_input('Số lần xác nhận trước khi giữ box bị bỏ sót',
                min_value=2, max_value=20, value=saved_memory['stable_hits'], key=f'adapt_memory_hits_{pid}', disabled=active))
            memory_policy['keep_misses'] = int(right.number_input('Số vòng bỏ sót được giữ nhãn dương',
                min_value=0, max_value=saved_memory['expire_misses']-1, value=saved_memory['keep_misses'],
                key=f'adapt_memory_misses_{pid}', disabled=active))
            st.caption('Ghép box trong cùng frame/class bằng IoU. Mặc định: ổn định ≥ 2 lần, giữ qua 1 lần '
                       'bỏ sót nếu còn ≥ 5 điểm, lần thứ 2 chuyển ignore, lần thứ 3 hết hạn. '
                       'Bằng chứng mâu thuẫn chuyển ignore ngay. Nhãn giữ theo lịch sử không được tính thêm vào agreement.')
            st.caption('Áp dụng khi bắt đầu lượt chạy mới. Snapshot và checkpoint cũ được giữ nguyên.')
        if st.button('Lưu thiết lập adaptation', disabled=active, type='primary'):
            adaptation.save_setup(store, pid, settings, selected, confirmed, int(bin_columns), int(epochs),
                                  learning_rate, source_weight, adaptation_classes=chosen_classes, memory_policy=memory_policy)
            st.rerun()
    st.subheader('3. Kiểm tra và bắt đầu')
    dirty = (any(selected[k] != config['models'].get(k, {}).get('path') for k in ('A', 'B'))
             or confirmed != config.get('coordinates_confirmed', False)
             or int(bin_columns) != config.get('bin_columns', 4)
             or int(epochs) != config.get('epochs', 1)
             or learning_rate != config.get('learning_rate', .0001)
             or source_weight != config.get('source_weight', .5)
             or set(chosen_classes) != set(saved_scope) or memory_policy != saved_memory)
    if dirty:
        st.info('Lưu thiết lập vừa thay đổi trước khi kiểm tra hoặc chạy.')
    if check.get('calibration_reports'):
        st.markdown('**Dự đoán trên source validation**')
        rows = []
        for key, report in check['calibration_reports'].items():
            for row in report['classes']:
                rows.append({'Detector': key, 'Class': row['class_name'],
                             'Số box dự đoán': row['samples'], 'Tối thiểu': report['minimum_samples'],
                             'Box có IoU ≥ 0.3': row['matched_at_iou_0_3'],
                             'Calibration': 'Đủ số mẫu' if row['enough_samples'] else 'Thiếu mẫu'})
        st.dataframe(rows, hide_index=True, width='stretch')
        st.caption('Chỉ dùng nhãn source validation, không dùng gold target. Đây là số box dự đoán, '
                   'không phải số nhãn có sẵn. Đủ số mẫu chưa chứng minh model dự đoán tốt.')
    if check['blockers']:
        for blocker in check['blockers']:
            st.write('• ' + blocker)
    else:
        st.success('Hai adapter đã qua kiểm tra GPU; chưa ghi nhận class thiếu mẫu calibration.')
    a, b = st.columns(2)
    if a.button('Kiểm tra adapter GPU A → B', disabled=active or dirty or not check['can_check'],
                help='Inference source/target và forward/backward cho mỗi model, không optimizer step và không cập nhật weights.'):
        with st.spinner('Kiểm tra môi trường và gửi job GPU…'):
            adaptation.start_job(store, pid, settings, mode='verify')
        st.rerun()
    next_error = None
    retry = False
    try:
        next_round = store.next_round(pid)
        if next_round and (store.path(pid) / 'work' / f'M{next_round}').exists():
            retry = True
    except ValueError as error:
        next_round = None
        next_error = str(error)
    if next_error:
        st.info(next_error)
    recoverable = resume_options(store, pid) if retry and not active else []
    if recoverable:
        lookup = {p['work']: p for p in recoverable}
        picked = st.selectbox('Lượt cần tiếp tục', list(lookup),
            format_func=lambda path: Path(path).name[:8] + ' · đã train xong ' +
                (', '.join(lookup[path]['completed']) or 'chưa model nào'), key='resume_attempt_' + pid)
        plan = lookup[picked]
        if len(plan['completed']) == 2:
            st.info(f'A/B của lượt {Path(picked).name[:8]} đã train xong. Tiếp tục sẽ dùng checkpoint đã lưu, '
                    'hoàn tất predict còn thiếu và hợp nhất nhãn; không train lại A/B.')
        else:
            done = ', '.join(plan['completed']) or 'Chưa có model hoàn tất'
            st.info('Giữ kết quả: ' + done + '. Model còn dở dùng checkpoint gần nhất nếu có; '
                    'nếu chưa có checkpoint target thì train lại riêng model đó từ model khởi tạo.')
        st.caption('Tạo lượt mới, giữ artifacts cũ; frame predict đã lưu và đúng hash được dùng lại. '
                   'Sau vòng này, tool tiếp tục theo chế độ chạy và chọn nhãn của dự án: '
                   'tự động khi đủ điều kiện, dừng chờ bạn khi chọn thủ công hoặc chưa có nhãn được chấp nhận.')
        if st.button(f'Tiếp tục vòng M{next_round}', type='primary',
                     disabled=dirty or not check['ready'] or next_round is None):
            with st.spinner('Kiểm tra artifacts và gửi phần công việc còn thiếu…'):
                adaptation.start_job(store, pid, settings, resume_from=picked)
            st.rerun()
    if retry and not active:
        st.caption(f'“Chạy lại vòng M{next_round}” bên dưới bắt đầu toàn bộ vòng từ checkpoint khởi tạo. '
                   'Dùng “Tiếp tục” để giữ công việc đã hoàn tất.' if recoverable else
            f'Vòng M{next_round} trước đã bị ngắt. Chạy lại sẽ tạo lượt mới và giữ toàn bộ log/checkpoint cũ. '
                'Các bước predict và adaptation của vòng này bắt đầu lại; model nguồn và kết quả kiểm tra GPU vẫn được giữ.')
    if b.button(f'Chạy lại vòng M{next_round}' if retry else 'Chạy vòng tiếp theo',
                type='secondary' if recoverable else 'primary',
                disabled=active or dirty or not check['ready'] or next_round is None or bool(next_error)):
        with st.spinner('Kiểm tra checkpoint/dataset và gửi controller…'):
            if retry:
                adaptation.start_job(store, pid, settings, retry=True)
            else:
                adaptation.start_job(store, pid, settings)
        st.rerun()
    if project['stop_after_round']:
        st.info('Đang bật “Kết thúc sau vòng hiện tại”, nên chưa thể gửi vòng mới. '
                'Cho phép chạy tiếp để mở lại nút chạy; giữ tất cả phiên bản đã lưu.')
        if st.button('Cho phép chạy tiếp', disabled=active,
                     help='Bỏ yêu cầu kết thúc. Không tự khởi chạy model; sau đó bạn bấm nút chạy vòng.'):
            store.allow_continue(pid)
            st.rerun()
    elif st.button('Kết thúc sau vòng hiện tại'):
        store.request_finish(pid)
        st.rerun()
    st.caption('Thay đổi số vòng/chế độ chạy tại Thiết lập dự án ở sidebar. '
               'Không dùng gold target. Hai model không có đủ bằng chứng chung sẽ dừng trước khi train.')
    monitor(store.path(pid))


def monitor(project_root):
    waiting = st.session_state.get(f'adaptation_waiting_{project_root.name}')
    jobs = (project_root / 'adaptation_jobs').glob('*/state.json')
    live = bool(waiting) or any(adaptation.job_status(p.parent)['process_alive'] for p in jobs)
    _monitor(project_root, live=live)


@live_fragment(5)
def _monitor(project_root):
    waiting_key = f'adaptation_waiting_{project_root.name}'
    waiting = st.session_state.get(waiting_key)
    if waiting and not adaptation.job_status(waiting)['process_alive']:
        # The controls above are outside this five-second fragment. Refresh the
        # whole app once the GPU job exits so readiness/buttons are recomputed.
        st.session_state.pop(waiting_key, None)
        st.rerun(scope='app')
    paths = sorted((project_root / 'adaptation_jobs').glob('*/state.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        return
    st.subheader('Theo dõi adaptation')
    selected = st.selectbox('Job adaptation', [str(p.parent) for p in paths],
                            format_func=lambda p: read(Path(p) / 'spec.json')['mode'] + ' · ' + Path(p).name[:8])
    root = Path(selected)
    state = adaptation.job_status(root)
    if state['status'] == 'completed' and state['mode'] == 'verify':
        st.success('Đã kiểm tra forward/backward thành công cho A và B.')
    else:
        st.write('Trạng thái: **' + state['status'] + '** · ' + STAGES.get(state.get('stage'), 'Đang chuẩn bị'))
    if state.get('active_detector'):
        st.caption('Detector ' + state['active_detector'])
    if state.get('error'):
        st.error(state['error'])
    inference = inference_progress(root, state)
    if inference:
        if inference['total']:
            title = 'Predict target' if inference['kind'] == 'target' else 'Calibration source'
            st.progress(min(1., inference['completed'] / inference['total']),
                text=f"{title} · {inference['detector']} · {inference['completed']}/{inference['total']} frame")
        else:
            st.caption('Worker đang nạp model/biên dịch CUDA; chưa có frame predict hoàn tất.')
        if not inference['live']:
            st.caption('Tiến trình đã dừng. Tiến độ trên là kết quả đã lưu, không phải job đang chạy.')
    if state['mode'] == 'run' and st.button('Xem dự đoán A/B trước/sau'):
        st.session_state['pending_navigation'] = 'So sánh phiên bản'
        st.rerun(scope='app')
    if st.button('Ngắt job ngay', disabled=not state['process_alive'],
                 help='Dừng controller và worker hiện tại; giữ log/checkpoint. Chưa tự resume vòng bị ngắt.'):
        adaptation.stop_job(root)
        st.rerun()
    if state['mode'] == 'run':
        work = Path(state['work_directory']) if state.get('work_directory') else None
        runs = sorted((work.glob('[AB]/progress.json') if work else (project_root / 'work').glob('M*/**/[AB]/progress.json')),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            path = runs[0]
            progress = read(path)
            epochs = max(int(progress.get('epochs') or 1), 1)
            # Native trainers write epoch_completed; older MMDet workers write the running epoch.
            done = int(progress.get('epoch_completed', max(int(progress.get('epoch', 1)) - 1, 0)))
            label = f"epoch {done}/{epochs}" + (f" · đang chạy epoch {progress['epoch']}" if 'epoch' in progress and progress.get('status') == 'running' else '')
            st.progress(min(1., done / epochs),
                        text=f"{path.relative_to(project_root / 'work').parts[0]} · {path.parent.name} · {label}")
            a, b, c = st.columns(3)
            a.metric('Optimizer steps', progress.get('optimizer_steps', 0))
            b.metric('Source loss', f"{progress.get('source_loss', 0):.4f}")
            c.metric('Target loss', f"{progress.get('target_loss', 0):.4f}")
            st.code(tail(path.parent / 'train.log'), language='log', height=180)
    with st.expander('Log controller & worker', expanded=state['status'] == 'failed'):
        st.code(tail(root / 'controller.log'), language='log', height=180)
        logs = sorted(root.glob('**/*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs and logs[0] != root / 'controller.log':
            st.caption(str(logs[0].relative_to(root)))
            st.code(tail(logs[0]), language='log', height=180)
