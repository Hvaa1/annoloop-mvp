"""Read-only memory inspection; no model or target-label imports."""
import streamlit as st
from annoloop.io import read, digest


def memory_view(store, project):
    root = store.path(project['id']) / 'rounds'
    versions = [v for v in project['completed_rounds'] if read(root / v / 'round.json').get('memory_sha256')]
    with st.expander('Bộ nhớ pseudo-label qua các vòng', expanded=bool(versions)):
        if not versions:
            st.info('Các vòng đã lưu chưa có snapshot bộ nhớ. Lượt chạy mới sẽ dựng lịch sử từ nhãn tự động '
                    'cũ và lưu bộ nhớ riêng; không thay đổi kết quả cũ.')
            return
        version = st.selectbox('Vòng có bộ nhớ', versions, index=len(versions)-1, key='memory_version_'+project['id'])
        receipt = read(root / version / 'round.json')
        path = root / version / 'memory.json'
        if digest(path) != receipt['memory_sha256']:
            st.error('Snapshot bộ nhớ đã thay đổi; giữ artifacts và kiểm tra trước khi dùng.')
            return
        memory = read(path)
        stats = memory['summary']
        for col, title, value in zip(st.columns(4), ['Quan sát hiện tại','Giữ nhờ lịch sử','Ignore','Hết hạn'],
                                    [stats['observed'],stats['retained'],stats['ignored'],stats['expired']]):
            col.metric(title,value)
        st.caption('Memory ID theo một box trong cùng frame qua nhiều phiên bản model; không phải track ID giữa các frame. '
                   'Vòng bị từ chối vẫn lưu quan sát để theo dõi; nhãn train chỉ lấy từ phiên bản đã chọn.')
        frames = {f['frame_id']: f['objects'] for f in memory['frames'] if f['objects']}
        if not frames:
            st.info('Chưa có box được ghi vào bộ nhớ.')
            return
        fid = st.selectbox('Frame trong bộ nhớ', list(frames), key='memory_frame_'+project['id'])
        objects = {e['id']: e for e in frames[fid]}
        identity = st.selectbox('Box cần theo dõi', list(objects), key='memory_box_'+project['id'],
                               format_func=lambda key: objects[key]['class_name']+' · '+key[-8:])
        entry = objects[identity]
        st.write(f"Trạng thái: {entry['state']} · {entry['reason']} · xác nhận {entry['hit_count']} "
                 f"· ổn định {entry['stable_hits']} · bỏ sót {entry['miss_count']} · mâu thuẫn {entry['conflict_count']}")
        st.dataframe([{'Vòng': 'M'+str(h['round_index']), 'Nguồn': h['origin'],
                       'Confidence': h['confidence'], 'Quality': h['quality'], 'IoU lịch sử': h['match_iou'],
                       'Số điểm': h['point_count'], 'Quan sát mới': h['independent_evidence'],
                       'Kết quả': h['reason']} for h in entry['history']], hide_index=True, width='stretch')
        st.caption('Giá trị trống nghĩa là không có box hiện tại để đo; điểm lịch sử không được biến thành dự đoán mới.')
