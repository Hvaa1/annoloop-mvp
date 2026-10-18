"""Explain saved automatic selection decisions without dispatching any job."""
import streamlit as st
from annoloop.io import read


REASONS = {
    'no_boxes': 'Không có nhãn chung để train',
    'agreement_below_threshold': 'Tỷ lệ đồng thuận thấp hơn ngưỡng',
    'supported_fraction_below_threshold': 'Tỷ lệ box đủ điểm thấp hơn ngưỡng',
    'agreement_regressed': 'Tỷ lệ đồng thuận giảm quá mức cho phép',
    'supported_fraction_regressed': 'Tỷ lệ box đủ điểm giảm quá mức cho phép',
    'box_growth_exceeded': 'Số box tăng quá mức cho phép',
    'agreement_decreased': 'Tỷ lệ đồng thuận thấp hơn vòng đang dùng',
}


def agreement_rule_text(policy, decision):
    # Rounds decided before the trend rule carry no rule name: they used the absolute threshold.
    if (decision.get('agreement_rule') or 'threshold') == 'threshold':
        return f"≥ {policy['min_agreement']:.2%}"
    if decision.get('compared_with_previous'):
        return f"≥ vòng đang dùng ({decision['previous_agreement']:.2%})"
    return 'Mốc đầu (chưa có vòng cùng cách tính để so)'


def selection_status(store, project):
    versions = project['completed_rounds']
    if not versions:
        return
    version = versions[-1]
    receipt = read(store.path(project['id']) / 'rounds' / version / 'round.json')
    decision, metrics = receipt['decision'], receipt['metrics']
    selected = project['selected_version']
    action = decision['action']
    if action == 'reject':
        reasons = '; '.join(REASONS.get(r, r) for r in decision['reasons'])
        st.warning(f'{version} không được tự động chọn: {reasons}.')
        if selected == version:
            st.info(f'Bạn đã chọn {version} bằng tay sau khi tool từ chối tự động. Nhãn đang dùng: {selected}.')
        elif selected:
            st.info(f'Giữ nhãn {selected}. Tool không tự thay bằng nhãn {version} chưa đạt ngưỡng.')
        else:
            st.info('Chưa có phiên bản nhãn được chấp nhận để train vòng tiếp. '
                    'Xem sample và lý do từ chối trước khi quyết định dùng vòng này.')
    elif action == 'accept':
        st.success(f'Tool đã tự động chấp nhận {version}. Nhãn đang dùng: {selected or "Chưa chọn"}.')
    elif project.get('reviewed_round') != version:
        st.info(f'{version} đang chờ bạn duyệt vì chế độ chọn nhãn là thủ công.')
    else:
        st.info(f'Đã duyệt {version}. Nhãn đang dùng: {selected or "Chưa chọn"}.')
    with st.expander('Kết quả chọn nhãn và ngưỡng'):
        saved_policy = receipt.get('selection_policy')
        policy = saved_policy or project['selection_policy']
        st.dataframe([
            {'Tiêu chí': 'Đồng thuận', 'Kết quả': f"{metrics['agreement']:.2%}",
             'Ngưỡng': agreement_rule_text(policy, decision)},
            {'Tiêu chí': 'Box có đủ điểm', 'Kết quả': f"{metrics['supported_fraction']:.2%}",
             'Ngưỡng': f"≥ {policy['min_supported_fraction']:.2%}"},
            {'Tiêu chí': 'Nhãn chung', 'Kết quả': str(metrics['box_count']), 'Ngưỡng': '> 0'},
            *([{'Tiêu chí': 'Nhãn dùng để train (đủ điểm)', 'Kết quả': str(metrics['trained_box_count']),
                'Ngưỡng': f"≥ {metrics.get('min_positive_points', 5)} point / box"}] if 'trained_box_count' in metrics else []),
        ], hide_index=True, width='stretch')
        if metrics.get('agreement_definition'):
            st.caption(f"Đồng thuận = số cặp A/B được nhận ÷ số box confidence ≥ {metrics.get('agreement_confidence', .7):g} "
                       f"của model nhiều box hơn ở mỗi frame ({metrics.get('confident_box_denominator', 0)} box).")
        else:
            st.caption('Vòng cũ: đồng thuận chia cho mọi box thô (score ≥ 0.05), nên thường rất thấp; '
                       'không so trực tiếp với vòng dùng cách tính mới.')
        st.caption('Các chỉ số này là proxy trên target không nhãn, không đo accuracy hoặc recall. '
                   + ('Ngưỡng được lưu lúc hoàn tất vòng.' if saved_policy else
                      'Vòng cũ chưa lưu ngưỡng riêng; bảng dùng ngưỡng hiện tại của dự án.'))
        st.write(f"Giảm proxy tối đa: {policy['max_proxy_drop']:.2%}; tăng số box tối đa: {policy['max_box_growth']:g} lần.")
