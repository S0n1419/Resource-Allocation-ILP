import os
import joblib
import pandas as pd
import numpy as np
import pulp
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Optional

# Import chatbot utilities
from dss_chatbot import (
    RecruitmentConstraints,
    mock_parse_query,
    parse_query_with_gemini,
    format_bot_response,
    validate_and_fix
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Recruitment DSS - Chatbot Tối Ưu Hóa Nguồn Lực",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (CSS Injection)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Gradient */
    .gradient-text {
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    
    /* Card design */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
        margin-bottom: 1rem;
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #6366f1;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA & MODEL LOADERS (Cached) -----------------
@st.cache_data
def load_and_preprocess_data():
    csv_path = 'data/job_salary_prediction_dataset.csv'
    if not os.path.exists(csv_path):
        st.error(f"Không tìm thấy file dữ liệu {csv_path}!")
        return None, None
        
    df = pd.read_csv(csv_path)
    
    # Load ML Model
    model_path = 'data/best_salary_prediction_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy mô hình ML dự báo lương {model_path}!")
        return df, None
        
    best_model = joblib.load(model_path)
    
    # Compute Composite Quality Index (q)
    s = 10 * (df['skills_count'] - 1) / 18
    t = 10 * df['certifications'] / 5
    y = 10 * np.log1p(df['experience_years']) / np.log(21)
    
    edu_mapping = {'High School': 2, 'Diploma': 4, 'Bachelor': 6, 'Master': 8, 'PhD': 10}
    d = df['education_level'].map(edu_mapping)
    
    company_mapping = {'Startup': 4, 'Small': 5, 'Medium': 6, 'Large': 8, 'Enterprise': 10}
    f = df['company_size'].map(company_mapping)
    
    df['quality_score'] = 0.25 * s + 0.08 * t + 0.47 * y + 0.11 * d + 0.09 * f
    
    # Predict salaries
    X_features = df.drop('salary', axis=1)
    df['predicted_salary'] = best_model.predict(X_features)
    
    # Recruitment cost estimation p = 1.3 * predicted_salary
    alpha = 0.3
    df['recruitment_cost'] = (1 + alpha) * df['predicted_salary']
    
    return df, best_model

# ----------------- ILP SOLVER ENGINE -----------------
def solve_recruitment_ilp_engine(
    df: pd.DataFrame,
    constraints: RecruitmentConstraints
) -> Optional[pd.DataFrame]:
    """
    Executes the Integer Linear Programming solver using PuLP based on extracted constraints.
    """
    budget = constraints.budget
    headcount = constraints.num_employees
    allowed_titles = constraints.job_titles
    
    # 1. Hard filters
    filtered_df = df.copy()
    if allowed_titles:
        filtered_df = filtered_df[filtered_df['job_title'].isin(allowed_titles)]
    
    if getattr(constraints, 'allowed_experience_years', None) is not None:
        filtered_df = filtered_df[filtered_df['experience_years'].isin(constraints.allowed_experience_years)]
    else:
        if constraints.experience_min is not None:
            filtered_df = filtered_df[filtered_df['experience_years'] >= constraints.experience_min]
        if constraints.experience_max is not None:
            filtered_df = filtered_df[filtered_df['experience_years'] <= constraints.experience_max]
        
    if constraints.min_skill_count is not None:
        filtered_df = filtered_df[filtered_df['skills_count'] >= constraints.min_skill_count]
        
    if constraints.min_certifications is not None:
        filtered_df = filtered_df[filtered_df['certifications'] >= constraints.min_certifications]
        
    # Hard filter for allowed education levels (unconditional)
    if getattr(constraints, 'allowed_education_levels', None):
        filtered_df = filtered_df[filtered_df['education_level'].isin(constraints.allowed_education_levels)]
    elif constraints.education_levels and constraints.education_ratio_min is None and constraints.education_ratio_max is None:
        # Backward compatibility fallback
        filtered_df = filtered_df[filtered_df['education_level'].isin(constraints.education_levels)]

    # Hard filter for allowed remote types (unconditional)
    remote_map = {'Remote': 'Yes', 'On-site': 'No', 'Hybrid': 'Hybrid'}
    if getattr(constraints, 'allowed_remote_types', None):
        db_allowed_remote = [remote_map[r] for r in constraints.allowed_remote_types if r in remote_map]
        if db_allowed_remote:
            filtered_df = filtered_df[filtered_df['remote_work'].isin(db_allowed_remote)]
    elif constraints.remote_types and constraints.remote_ratio_min is None and constraints.remote_ratio_max is None:
        # Backward compatibility fallback
        db_remote_types = [remote_map[r] for r in constraints.remote_types if r in remote_map]
        if db_remote_types:
            filtered_df = filtered_df[filtered_df['remote_work'].isin(db_remote_types)]

    # Map candidate experience groups
    def get_exp_group(years):
        if years < 1:
            return 'Fresher'
        elif years < 3:
            return 'Junior'
        elif years < 5:
            return 'Mid'
        elif years < 8:
            return 'Senior'
        else:
            return 'Expert'
            
    filtered_df['exp_group'] = filtered_df['experience_years'].apply(get_exp_group)
    
    # Sort by quality score and limit pool size to prevent solver crashes (out of memory)
    MAX_CANDIDATES = 5000
    if len(filtered_df) > MAX_CANDIDATES:
        filtered_df = filtered_df.sort_values(by="quality_score", ascending=False).head(MAX_CANDIDATES)
        
    filtered_df = filtered_df.reset_index(drop=True)
    n_candidates = len(filtered_df)
    
    if n_candidates == 0:
        return None
        
    # 2. Problem Initialization
    prob = pulp.LpProblem('Recruitment_Optimization', pulp.LpMaximize)
    
    # Decision variables
    x = [pulp.LpVariable(f'x_{h}', cat='Binary') for h in range(n_candidates)]
    
    # Objective function: Maximize quality
    prob += pulp.lpSum(filtered_df.loc[h, 'quality_score'] * x[h] for h in range(n_candidates)), 'Total_Quality_Score'
    
    # Budget constraint
    prob += pulp.lpSum(filtered_df.loc[h, 'recruitment_cost'] * x[h] for h in range(n_candidates)) <= budget, 'Recruitment_Budget'
    
    # Headcount constraint
    prob += pulp.lpSum(x[h] for h in range(n_candidates)) == headcount, 'Headcount_Limit'
    
    # Job title bounds (at least 1 of each allowed title if possible, or user specified)
    if allowed_titles:
        for title in allowed_titles:
            title_indices = filtered_df[filtered_df['job_title'] == title].index
            if len(title_indices) > 0:
                prob += pulp.lpSum(x[h] for h in title_indices) >= 1, f'Default_Min_{title.replace(" ", "_")}'
                
    # Absolute limits per job title (job_title_limits)
    if constraints.job_title_limits:
        for title, limits in constraints.job_title_limits.items():
            title_indices = filtered_df[filtered_df['job_title'] == title].index
            if len(title_indices) > 0:
                if limits.min is not None:
                    prob += pulp.lpSum(x[h] for h in title_indices) >= limits.min, f'Min_Limit_{title.replace(" ", "_")}'
                if limits.max is not None:
                    prob += pulp.lpSum(x[h] for h in title_indices) <= limits.max, f'Max_Limit_{title.replace(" ", "_")}'

    # Job title max ratio limit (job_title_max_ratio)
    if constraints.job_title_max_ratio:
        for title, max_ratio in constraints.job_title_max_ratio.items():
            if max_ratio is not None:
                title_indices = filtered_df[filtered_df['job_title'] == title].index
                prob += pulp.lpSum(x[h] for h in title_indices) <= max_ratio * headcount, f'Max_Ratio_Title_{title.replace(" ", "_")}'
            
    # Experience structure constraints (experience_groups)
    if constraints.experience_groups:
        for grp_name in ['Fresher', 'Junior', 'Mid', 'Senior', 'Expert']:
            grp_constraints = getattr(constraints.experience_groups, grp_name, None)
            if grp_constraints:
                grp_indices = filtered_df[filtered_df['exp_group'] == grp_name].index
                if grp_constraints.min_ratio is not None:
                     prob += pulp.lpSum(x[h] for h in grp_indices) >= grp_constraints.min_ratio * headcount, f'Min_Ratio_{grp_name}'
                if grp_constraints.max_ratio is not None:
                     prob += pulp.lpSum(x[h] for h in grp_indices) <= grp_constraints.max_ratio * headcount, f'Max_Ratio_{grp_name}'
                if grp_constraints.min_count is not None:
                     prob += pulp.lpSum(x[h] for h in grp_indices) >= grp_constraints.min_count, f'Min_Count_{grp_name}'
                if grp_constraints.max_count is not None:
                     prob += pulp.lpSum(x[h] for h in grp_indices) <= grp_constraints.max_count, f'Max_Count_{grp_name}'
        
    # Education level constraint
    if constraints.education_levels and (constraints.education_ratio_min is not None or constraints.education_ratio_max is not None):
        edu_indices = filtered_df[filtered_df['education_level'].isin(constraints.education_levels)].index
        if constraints.education_ratio_min is not None:
            prob += pulp.lpSum(x[h] for h in edu_indices) >= constraints.education_ratio_min * headcount, 'Min_Education_Ratio'
        if constraints.education_ratio_max is not None:
            prob += pulp.lpSum(x[h] for h in edu_indices) <= constraints.education_ratio_max * headcount, 'Max_Education_Ratio'
            
    # Skills high constraints
    if constraints.skills_high and constraints.skills_high.threshold is not None:
        skills_high_indices = filtered_df[filtered_df['skills_count'] >= constraints.skills_high.threshold].index
        if constraints.skills_high.min_ratio is not None:
            prob += pulp.lpSum(x[h] for h in skills_high_indices) >= constraints.skills_high.min_ratio * headcount, 'Min_Skills_High_Ratio'
        if constraints.skills_high.max_ratio is not None:
            prob += pulp.lpSum(x[h] for h in skills_high_indices) <= constraints.skills_high.max_ratio * headcount, 'Max_Skills_High_Ratio'

    # Certifications high constraints
    if constraints.certifications_high and constraints.certifications_high.threshold is not None:
        certs_high_indices = filtered_df[filtered_df['certifications'] >= constraints.certifications_high.threshold].index
        if constraints.certifications_high.min_ratio is not None:
            prob += pulp.lpSum(x[h] for h in certs_high_indices) >= constraints.certifications_high.min_ratio * headcount, 'Min_Certs_High_Ratio'
        if constraints.certifications_high.max_ratio is not None:
            prob += pulp.lpSum(x[h] for h in certs_high_indices) <= constraints.certifications_high.max_ratio * headcount, 'Max_Certs_High_Ratio'
        
    # Work mode constraints
    if constraints.remote_types and (constraints.remote_ratio_min is not None or constraints.remote_ratio_max is not None):
        db_remote_types = [remote_map[r] for r in constraints.remote_types if r in remote_map]
        if db_remote_types:
            remote_indices = filtered_df[filtered_df['remote_work'].isin(db_remote_types)].index
            if constraints.remote_ratio_min is not None:
                prob += pulp.lpSum(x[h] for h in remote_indices) >= constraints.remote_ratio_min * headcount, 'Min_Remote_Ratio'
            if constraints.remote_ratio_max is not None:
                prob += pulp.lpSum(x[h] for h in remote_indices) <= constraints.remote_ratio_max * headcount, 'Max_Remote_Ratio'
        
    # Minimum average quality constraint
    if constraints.min_avg_quality_score is not None:
        prob += pulp.lpSum(filtered_df.loc[h, 'quality_score'] * x[h] for h in range(n_candidates)) >= constraints.min_avg_quality_score * headcount, 'Min_Avg_Quality_Score'
        
    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    
    if pulp.LpStatus[status] == 'Optimal':
        selected_indices = [h for h in range(n_candidates) if pulp.value(x[h]) > 0.5]
        return filtered_df.iloc[selected_indices].copy()
    return None

# Load data and ML model
df, model = load_and_preprocess_data()
if df is not None and 'id' not in df.columns:
    df['id'] = df.index

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/bot.png", width=80)
    st.markdown("<h2 style='color:#6366f1;margin-top:0;margin-bottom:1rem;'>Cấu Hình Hệ Thống</h2>", unsafe_allow_html=True)
    
    # Advanced Settings (API Settings & Dataset statistics collapsed by default)
    with st.expander("⚙️ Cấu Hình API & Dữ Liệu", expanded=False):
        # API Settings
        st.markdown("#### 🔑 API Key Settings")
        api_key = st.text_input("Gemini API Key", type="password", help="Lấy API Key từ Google AI Studio")
        use_mock = st.checkbox("Sử dụng Mock Parser (Không cần API Key)", value=True, help="Sử dụng regex để trích xuất nhanh các tham số chính bằng tiếng Việt")
        
        st.markdown("---")
        
        # Dataset statistics
        st.markdown("#### 📊 Thông Tin Dataset")
        if df is not None:
            st.write(f"- Tổng số ứng viên: **{len(df):,}**")
            st.write(f"- Chức danh: **{df['job_title'].nunique()} vị trí**")
            st.write(f"- ML Model: **Random Forest (Loaded)**")
            
    # Reset Chat
    st.markdown("---")
    if st.button("Reset Cuộc Trò Chuyện", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------- MAIN TITLE & MODE SELECTOR -----------------
st.markdown("<div class='gradient-text'>Hệ Thống Ra Quyết Định Phân Bổ Nhân Sự (ILP DSS)</div>", unsafe_allow_html=True)

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Hội thoại Chatbot"

# Clear, prominent side-by-side mode switching buttons
col_mode1, col_mode2 = st.columns(2)
with col_mode1:
    is_chat = st.session_state.app_mode == "Hội thoại Chatbot"
    if st.button(
        "🤖 HỘI THOẠI CHATBOT",
        use_container_width=True,
        type="primary" if is_chat else "secondary",
        help="Sử dụng AI dịch câu hỏi tự nhiên bằng tiếng Việt để tự động lập cấu hình tối ưu."
    ):
        st.session_state.app_mode = "Hội thoại Chatbot"
        st.rerun()

with col_mode2:
    is_manual = st.session_state.app_mode == "Nhập thông số trực tiếp"
    if st.button(
        "⚙️ NHẬP THÔNG SỐ THỦ CÔNG",
        use_container_width=True,
        type="primary" if is_manual else "secondary",
        help="Tự cấu hình biểu mẫu ràng buộc và chạy tối ưu hóa, quét độ nhạy, What-if hoặc Monte Carlo."
    ):
        st.session_state.app_mode = "Nhập thông số trực tiếp"
        st.rerun()

app_mode = st.session_state.app_mode
st.markdown("<hr style='margin-top:0.5rem;margin-bottom:1.5rem;border-color:rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

if app_mode == "Hội thoại Chatbot":
    st.write(
        "Chào mừng bạn đến với DSS thông minh. Nhập yêu cầu nhân sự bằng ngôn ngữ tự nhiên để hệ thống trích xuất tham số tự động, "
        "thiết lập bài toán Quy hoạch nguyên (ILP) và đề xuất đội hình tuyển dụng tối ưu nhất."
    )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Render visualizations if the message contains them
            if message["role"] == "assistant" and "results" in message:
                results_df = pd.read_json(message["results"])
                
                # Metric Columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Tổng Chi Phí</div>
                        <div class="metric-value">${results_df["recruitment_cost"].sum():,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Chất Lượng TB</div>
                        <div class="metric-value">{results_df["quality_score"].mean():.2f}/10</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Số Lượng Tuyển</div>
                        <div class="metric-value">{len(results_df)} người</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    budget = message["budget"]
                    utilization = (results_df["recruitment_cost"].sum() / budget) * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Hiệu Suất Sử Dụng Ngân Sách</div>
                        <div class="metric-value">{utilization:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Interactive Charts
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    # Job title distribution
                    fig_title = px.pie(
                        results_df, 
                        names="job_title", 
                        title="Cơ cấu Chức danh Tuyển dụng",
                        color_discrete_sequence=['#6366f1', '#a855f7', '#ec4899', '#3b82f6', '#10b981']
                    )
                    fig_title.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                    st.plotly_chart(fig_title, use_container_width=True, key=f"chat_hist_title_{idx}")
                with chart_col2:
                    # Experience distribution
                    fig_exp = px.histogram(
                        results_df, 
                        x="experience_years", 
                        nbins=5, 
                        title="Phân bố số năm kinh nghiệm tuyển được",
                        color_discrete_sequence=['#a855f7']
                    )
                    fig_exp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                    st.plotly_chart(fig_exp, use_container_width=True, key=f"chat_hist_exp_{idx}")
                    
                # Candidate Data Table
                st.markdown("### 📋 Danh sách ứng viên được đề xuất tuyển dụng")
                results_show = results_df[['job_title', 'experience_years', 'education_level', 'remote_work', 'company_size', 'recruitment_cost', 'quality_score']].copy()
                results_show.index = range(1, len(results_show) + 1)
                results_show.index.name = "STT"
                st.dataframe(
                    results_show,
                    use_container_width=True
                )
    
    # React to user input
    if prompt := st.chat_input("Nhập yêu cầu tuyển dụng (Ví dụ: tuyển 10 người, ngân sách 1.5 triệu đô, ít nhất 30% senior)"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate bot response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # 1. Parse parameters from user query
            if use_mock:
                message_placeholder.markdown("🔄 *Đang phân tích cú pháp câu hỏi bằng Mock Regex Parser...*")
                extracted_params = mock_parse_query(prompt)
            else:
                if not api_key:
                    st.warning("⚠️ Vui lòng cung cấp Gemini API Key trong thanh cấu hình hoặc tick vào 'Sử dụng Mock Parser'!")
                    extracted_params = None
                else:
                    message_placeholder.markdown("🔄 *Đang gửi yêu cầu phân tích ngữ nghĩa đến Gemini API...*")
                    try:
                        extracted_params = parse_query_with_gemini(prompt, api_key)
                    except Exception as e:
                        st.error(f"Lỗi khi gọi Gemini API: {str(e)}")
                        st.info("Hệ thống tự động chuyển sang sử dụng Mock Regex Parser để hỗ trợ.")
                        extracted_params = mock_parse_query(prompt)
            
            if extracted_params is not None:
                # Post-processing & Validation
                extracted_params, error_msg = validate_and_fix(extracted_params)
                
                if error_msg:
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    message_placeholder.markdown("🧠 *Đang thiết lập bài toán tối ưu và chạy ILP Solver (PuLP)...*")
                    
                    # Run the ILP solver
                    results_df = solve_recruitment_ilp_engine(df, extracted_params)
                    if results_df is not None:
                        st.session_state.base_optimal_team = results_df
                        st.session_state.base_manual_params = extracted_params
                    
                    # Format markdown response
                    bot_text = format_bot_response(
                        prompt, 
                        extracted_params, 
                        results_df, 
                        extracted_params.budget, 
                        extracted_params.num_employees
                    )
                    
                    message_placeholder.markdown(bot_text)
                    
                    # Store assistant response details in session state
                    message_data = {
                        "role": "assistant",
                        "content": bot_text
                    }
                    
                    # If solution is found, generate charts
                    if results_df is not None:
                        # Metric Columns
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Tổng Chi Phí</div>
                                <div class="metric-value">${results_df["recruitment_cost"].sum():,.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Chất Lượng TB</div>
                                <div class="metric-value">{results_df["quality_score"].mean():.2f}/10</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Số Lượng Tuyển</div>
                                <div class="metric-value">{len(results_df)} người</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col4:
                            budget = extracted_params.budget
                            utilization = (results_df["recruitment_cost"].sum() / budget) * 100
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Hiệu Suất Sử Dụng Ngân Sách</div>
                                <div class="metric-value">{utilization:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Render charts
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            fig_title = px.pie(
                                results_df, 
                                names="job_title", 
                                title="Cơ cấu Chức danh Tuyển dụng",
                                color_discrete_sequence=['#6366f1', '#a855f7', '#ec4899', '#3b82f6', '#10b981']
                            )
                            fig_title.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                            st.plotly_chart(fig_title, use_container_width=True, key=f"chat_new_title_{len(st.session_state.messages)}")
                        with chart_col2:
                            fig_exp = px.histogram(
                                results_df, 
                                x="experience_years", 
                                nbins=5, 
                                title="Phân bố số năm kinh nghiệm tuyển được",
                                color_discrete_sequence=['#a855f7']
                            )
                            fig_exp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                            st.plotly_chart(fig_exp, use_container_width=True, key=f"chat_new_exp_{len(st.session_state.messages)}")
                            
                        st.markdown("### 📋 Danh sách ứng viên được đề xuất tuyển dụng")
                        results_show = results_df[['job_title', 'experience_years', 'education_level', 'remote_work', 'company_size', 'recruitment_cost', 'quality_score']].copy()
                        results_show.index = range(1, len(results_show) + 1)
                        results_show.index.name = "STT"
                        st.dataframe(
                            results_show,
                            use_container_width=True
                        )
                        
                        # Save results to session state for persistence
                        message_data["results"] = results_df.to_json()
                        message_data["budget"] = extracted_params.budget
                    
                    st.session_state.messages.append(message_data)

else:  # app_mode == "Nhập thông số trực tiếp"
    st.write(
        "Cấu hình các tiêu chí và chạy tối ưu hóa tuyển dụng nhân sự hoặc chạy khảo sát độ nhạy của mô hình toán học."
    )
    
    # 3 tabs layout
    tabs = st.tabs(["🎯 Tối Ưu Hóa Nguồn Lực", "📈 Phân Tích Độ Nhạy", "💡 Kịch Bản Giả Định & Báo Cáo", "🎲 Mô Phỏng Monte Carlo"])
    
    with tabs[0]:
        col_input, col_output = st.columns([1.1, 1.4])
        
        with col_input:
            st.markdown("### ⚙️ Biểu Mẫu Ràng Buộc")
            with st.form("manual_constraints_form"):
                # 1. Basic Constraints
                budget = st.number_input("Ngân sách tối đa (USD)", min_value=10000.0, value=1500000.0, step=50000.0, help="Tổng chi phí tối đa (B)", key="man_budget")
                num_employees = st.number_input("Số lượng cần tuyển (người)", min_value=1, value=10, step=1, help="Tổng headcount (N)", key="man_employees")
                job_titles = st.multiselect(
                    "Các chức danh được phép tuyển",
                    ["AI Engineer", "Data Scientist", "Data Analyst", "Backend Developer", "Frontend Developer", "Machine Learning Engineer"],
                    default=["AI Engineer", "Data Scientist", "Data Analyst"],
                    help="Để trống để cho phép tất cả chức danh",
                    key="man_job_titles"
                )
                
                # Advanced nested expander inside the form
                with st.expander("⚙️ Cấu hình Ràng buộc Nâng cao (Không bắt buộc)"):
                    # 2. Experience & Quality
                    st.markdown("##### 🕒 Yêu cầu Kinh nghiệm & Chất lượng")
                    allowed_experience_years = st.multiselect(
                        "Số năm kinh nghiệm được chấp nhận (J^allow)",
                        options=list(range(21)),
                        default=list(range(21)),
                        help="Chỉ tuyển dụng ứng viên có số năm kinh nghiệm thuộc danh sách được chọn.",
                        key="man_allowed_exp"
                    )
                        
                    min_avg_quality_score = st.slider("Điểm chất lượng trung bình tối thiểu (0.0 - 10.0)", 0.0, 10.0, 0.0, step=0.1, key="man_avg_quality")
                    
                    # 3. Education
                    st.markdown("##### 🎓 Ràng buộc học vấn")
                    allowed_education_levels = st.multiselect(
                        "Trình độ học vấn được phép tuyển (Lọc cứng)",
                        ["High School", "Diploma", "Bachelor", "Master", "PhD"],
                        default=["Bachelor", "Master", "PhD"],
                        help="Chỉ cho phép tuyển ứng viên thuộc các trình độ này. Ứng viên có trình độ khác sẽ bị loại bỏ hoàn toàn.",
                        key="man_allowed_edu"
                    )
                    education_levels = st.multiselect(
                        "Nhóm trình độ học vấn để tính tỷ lệ (Tùy chọn)",
                        ["High School", "Diploma", "Bachelor", "Master", "PhD"],
                        default=[],
                        help="Chọn các trình độ để áp dụng ràng buộc tỷ lệ (ví dụ: Chọn Master và PhD để yêu cầu ít nhất 30% Master/PhD).",
                        key="man_edu_levels"
                    )
                    edu_col1, edu_col2 = st.columns(2)
                    with edu_col1:
                        education_ratio_min = st.slider("Tỷ lệ học vấn tối thiểu", 0.0, 1.0, 0.0, help="Tỷ lệ tối thiểu của nhân viên thuộc nhóm trình độ tính tỷ lệ ở trên", key="man_edu_ratio_min")
                    with edu_col2:
                        education_ratio_max = st.slider("Tỷ lệ học vấn tối đa", 0.0, 1.0, 1.0, help="Tỷ lệ tối đa của nhân viên thuộc nhóm trình độ tính tỷ lệ ở trên", key="man_edu_ratio_max")
                    
                    # 4. Work Mode
                    st.markdown("##### 🏢 Hình thức làm việc")
                    allowed_remote_types = st.multiselect(
                        "Hình thức làm việc được phép tuyển (Lọc cứng)",
                        ["On-site", "Hybrid", "Remote"],
                        default=["Hybrid", "Remote"],
                        help="Chỉ cho phép tuyển các ứng viên có hình thức làm việc này.",
                        key="man_allowed_remote"
                    )
                    remote_types = st.multiselect(
                        "Nhóm hình thức làm việc để tính tỷ lệ (Tùy chọn)",
                        ["On-site", "Hybrid", "Remote"],
                        default=[],
                        help="Chọn các hình thức làm việc để áp dụng ràng buộc tỷ lệ (ví dụ: Chọn Remote để giới hạn tối đa 20% Remote).",
                        key="man_remote_types"
                    )
                    remote_col1, remote_col2 = st.columns(2)
                    with remote_col1:
                        remote_ratio_min = st.slider("Tỷ lệ tối thiểu làm việc", 0.0, 1.0, 0.0, help="Tỷ lệ tối thiểu của nhóm tính tỷ lệ làm việc ở trên", key="man_remote_ratio_min")
                    with remote_col2:
                        remote_ratio_max = st.slider("Tỷ lệ tối đa làm việc", 0.0, 1.0, 1.0, help="Tỷ lệ tối đa của nhóm tính tỷ lệ làm việc ở trên", key="man_remote_ratio_max")
                        
                    # 5. Skills & Certifications counts
                    st.markdown("##### 🛠 Yêu cầu Kỹ năng & Chứng chỉ")
                    sc_col1, sc_col2 = st.columns(2)
                    with sc_col1:
                        min_skill_count = st.number_input("Số kỹ năng tối thiểu", min_value=0, value=0, step=1, key="man_min_skill")
                    with sc_col2:
                        min_certifications = st.number_input("Số chứng chỉ tối thiểu", min_value=0, value=0, step=1, key="man_min_cert")
                    
                    st.markdown("##### 🕒 Cơ cấu nhóm kinh nghiệm")
                    exp_grps = {}
                    for grp in ['Fresher', 'Junior', 'Mid', 'Senior', 'Expert']:
                        st.markdown(f"**Nhóm {grp}**")
                        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
                        with g_col1:
                            g_min_r = st.number_input("Tỷ lệ min", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key=f"m_r_{grp}")
                        with g_col2:
                            g_max_r = st.number_input("Tỷ lệ max", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key=f"x_r_{grp}")
                        with g_col3:
                            g_min_c = st.number_input("SL min", min_value=0, value=0, step=1, key=f"m_c_{grp}")
                        with g_col4:
                            g_max_c = st.number_input("SL max", min_value=0, value=0, step=1, key=f"x_c_{grp}")
                            
                        grp_rules = {}
                        if g_min_r > 0: grp_rules['min_ratio'] = g_min_r
                        if g_max_r < 1.0: grp_rules['max_ratio'] = g_max_r
                        if g_min_c > 0: grp_rules['min_count'] = g_min_c
                        if g_max_c > 0: grp_rules['max_count'] = g_max_c
                        if grp_rules:
                            exp_grps[grp] = grp_rules
                            
                    st.markdown("##### 🚀 Ràng buộc kỹ năng cao (High Skills)")
                    s_thresh = st.number_input("Ngưỡng kỹ năng cao (số lượng)", min_value=0, value=12, step=1, key="man_s_thresh")
                    s_min_r = st.number_input("Tỷ lệ tối thiểu kỹ năng cao", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="man_s_min_r")
                    s_max_r = st.number_input("Tỷ lệ tối đa kỹ năng cao", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key="man_s_max_r")
                    skills_high = None
                    if s_min_r > 0 or s_max_r < 1.0:
                        skills_high = {'threshold': s_thresh, 'min_ratio': s_min_r if s_min_r > 0 else None, 'max_ratio': s_max_r if s_max_r < 1.0 else None}

                    st.markdown("##### 📜 Ràng buộc chứng chỉ cao (High Certifications)")
                    c_thresh = st.number_input("Ngưỡng chứng chỉ cao (số lượng)", min_value=0, value=4, step=1, key="man_c_thresh")
                    c_min_r = st.number_input("Tỷ lệ tối thiểu chứng chỉ cao", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="man_c_min_r")
                    c_max_r = st.number_input("Tỷ lệ tối đa chứng chỉ cao", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key="man_c_max_r")
                    certs_high = None
                    if c_min_r > 0 or c_max_r < 1.0:
                        certs_high = {'threshold': c_thresh, 'min_ratio': c_min_r if c_min_r > 0 else None, 'max_ratio': c_max_r if c_max_r < 1.0 else None}

                    st.markdown("##### 📊 Giới hạn số lượng & Tỷ lệ theo chức danh")
                    job_title_limits = {}
                    job_title_max_ratio = {}
                    active_titles = job_titles if job_titles else ["AI Engineer", "Data Scientist", "Data Analyst", "Backend Developer", "Frontend Developer", "Machine Learning Engineer"]
                    for title in active_titles:
                        st.markdown(f"**{title}**")
                        t_col1, t_col2, t_col3 = st.columns(3)
                        with t_col1:
                            t_min = st.number_input("SL tối thiểu", min_value=0, value=0, key=f"t_min_{title}")
                        with t_col2:
                            t_max = st.number_input("SL tối đa", min_value=0, value=0, key=f"t_max_{title}")
                        with t_col3:
                            t_ratio = st.number_input("Tỷ lệ tối đa", min_value=0.0, max_value=1.0, value=1.0, key=f"t_rat_{title}")
                            
                        if t_min > 0 or t_max > 0:
                            job_title_limits[title] = {'min': t_min if t_min > 0 else None, 'max': t_max if t_max > 0 else None}
                        if t_ratio < 1.0:
                            job_title_max_ratio[title] = t_ratio
                
                submit_button = st.form_submit_button("Chạy Tối Ưu Hóa 🚀", use_container_width=True)
                
        with col_output:
            st.markdown("### 📊 Kết Quả Tối Ưu Hóa")
            if submit_button:
                # Construct manual constraints model
                manual_params = RecruitmentConstraints(
                    budget=budget,
                    num_employees=num_employees,
                    job_titles=job_titles if job_titles else None,
                    job_title_limits=job_title_limits if job_title_limits else None,
                    job_title_max_ratio=job_title_max_ratio if job_title_max_ratio else None,
                    allowed_experience_years=allowed_experience_years if allowed_experience_years else None,
                    experience_groups=exp_grps if exp_grps else None,
                    allowed_education_levels=allowed_education_levels if allowed_education_levels else None,
                    education_levels=education_levels if education_levels else None,
                    education_ratio_min=education_ratio_min if education_ratio_min > 0.0 else None,
                    education_ratio_max=education_ratio_max if education_ratio_max < 1.0 else None,
                    min_skill_count=min_skill_count if min_skill_count > 0 else None,
                    skills_high=skills_high,
                    min_certifications=min_certifications if min_certifications > 0 else None,
                    certifications_high=certs_high,
                    allowed_remote_types=allowed_remote_types if allowed_remote_types else None,
                    remote_types=remote_types if remote_types else None,
                    remote_ratio_min=remote_ratio_min if remote_ratio_min > 0.0 else None,
                    remote_ratio_max=remote_ratio_max if remote_ratio_max < 1.0 else None,
                    min_avg_quality_score=min_avg_quality_score if min_avg_quality_score > 0.0 else None
                )
                
                # Post-processing validate
                manual_params, error_msg = validate_and_fix(manual_params)
                
                if error_msg:
                    st.error(error_msg)
                else:
                    with st.spinner("Đang chạy PuLP ILP solver..."):
                        results_df = solve_recruitment_ilp_engine(df, manual_params)
                        
                    if results_df is not None:
                        st.success("🎉 Đã tìm thấy phương án tuyển dụng tối ưu!")
                        st.session_state.base_optimal_team = results_df
                        st.session_state.base_manual_params = manual_params
                        
                        # Display metrics
                        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                        with m_col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Tổng Chi Phí</div>
                                <div class="metric-value">${results_df["recruitment_cost"].sum():,.2f}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m_col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Chất Lượng TB</div>
                                <div class="metric-value">{results_df["quality_score"].mean():.2f}/10</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m_col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Số Lượng Tuyển</div>
                                <div class="metric-value">{len(results_df)} người</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m_col4:
                            utilization = (results_df["recruitment_cost"].sum() / budget) * 100
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-title">Sử dụng ngân sách</div>
                                <div class="metric-value">{utilization:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Charts
                        ch_col1, ch_col2 = st.columns(2)
                        with ch_col1:
                            fig_title = px.pie(
                                results_df, 
                                names="job_title", 
                                title="Cơ cấu Chức danh",
                                color_discrete_sequence=['#6366f1', '#a855f7', '#ec4899', '#3b82f6', '#10b981']
                            )
                            fig_title.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                            st.plotly_chart(fig_title, use_container_width=True, key="manual_opt_title")
                        with ch_col2:
                            fig_exp = px.histogram(
                                results_df, 
                                x="experience_years", 
                                nbins=5, 
                                title="Phân bố số năm kinh nghiệm",
                                color_discrete_sequence=['#a855f7']
                            )
                            fig_exp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                            st.plotly_chart(fig_exp, use_container_width=True, key="manual_opt_exp")
                            
                        st.markdown("##### 📋 Chi tiết ứng viên đề xuất")
                        results_show = results_df[['job_title', 'experience_years', 'education_level', 'remote_work', 'company_size', 'recruitment_cost', 'quality_score']].copy()
                        results_show.index = range(1, len(results_show) + 1)
                        results_show.index.name = "STT"
                        st.dataframe(
                            results_show,
                            use_container_width=True
                        )
                    else:
                        st.error("❌ Không tìm được phương án khả thi! Vui lòng tăng ngân sách, giảm headcount hoặc nới lỏng các tiêu chí sàng lọc.")
            else:
                st.info("💡 Điền đầy đủ các thông số ở cột bên trái và click nút **Chạy Tối Ưu Hóa** để hiển thị kết quả phân bổ nhân sự.")

    with tabs[1]:
        st.markdown("### 📈 Phân Tích Độ Nhạy Một Chiều (One-Way Sensitivity Sweep)")
        st.write(
            "Khảo sát tác động của việc thay đổi một tham số đầu vào duy nhất lên tổng chất lượng "
            "và số lượng nhân sự tuyển dụng được, trong khi giữ cố định tất cả các ràng buộc khác."
        )
        
        # Sweep configurations
        col_sweep_cfg, col_sweep_charts = st.columns([1, 1.8])
        
        with col_sweep_cfg:
            st.markdown("#### ⚙️ Cấu Hình Khảo Sát")
            sweep_param = st.selectbox(
                "Chọn tham số khảo sát:",
                ["Ngân sách (Budget)", "Số lượng tuyển (Headcount)", "Tỷ lệ Senior tối thiểu", "Tỷ lệ Remote tối thiểu", "Hệ số chi phí lương (Salary Scale)"],
                key="sweep_param_selector"
            )
            
            # Set defaults based on parameter
            if sweep_param == "Ngân sách (Budget)":
                def_start = float(budget) * 0.6
                def_end = float(budget) * 1.4
                def_step = float(budget) * 0.1
                
                sweep_start = st.number_input("Từ ngân sách (USD)", min_value=10000.0, value=def_start, step=50000.0, key="sw_b_start")
                sweep_end = st.number_input("Đến ngân sách (USD)", min_value=10000.0, value=def_end, step=50000.0, key="sw_b_end")
                sweep_step = st.number_input("Bước nhảy (Step) (USD)", min_value=1000.0, value=def_step, step=10000.0, key="sw_b_step")
                
            elif sweep_param == "Số lượng tuyển (Headcount)":
                def_start = max(1, int(num_employees) - 5)
                def_end = int(num_employees) + 5
                def_step = 1
                
                sweep_start = st.number_input("Từ số lượng (người)", min_value=1, value=def_start, step=1, key="sw_h_start")
                sweep_end = st.number_input("Đến số lượng (người)", min_value=1, value=def_end, step=1, key="sw_h_end")
                sweep_step = st.number_input("Bước nhảy (Step) (người)", min_value=1, value=def_step, step=1, key="sw_h_step")
                
            elif sweep_param == "Tỷ lệ Senior tối thiểu":
                sweep_start = st.slider("Từ tỷ lệ Senior", 0.0, 1.0, 0.20, step=0.05, key="sw_s_start")
                sweep_end = st.slider("Đến tỷ lệ Senior", 0.0, 1.0, 0.60, step=0.05, key="sw_s_end")
                sweep_step = st.slider("Bước nhảy (Step)", 0.01, 0.20, 0.05, step=0.01, key="sw_s_step")
                
            elif sweep_param == "Tỷ lệ Remote tối thiểu":
                sweep_start = st.slider("Từ tỷ lệ Remote", 0.0, 1.0, 0.20, step=0.05, key="sw_r_start")
                sweep_end = st.slider("Đến tỷ lệ Remote", 0.0, 1.0, 0.80, step=0.05, key="sw_r_end")
                sweep_step = st.slider("Bước nhảy (Step)", 0.01, 0.20, 0.10, step=0.01, key="sw_r_step")
                
            else:  # Hệ số chi phí lương
                sweep_start = st.slider("Hệ số bắt đầu (scale factor)", 0.80, 1.20, 0.80, step=0.05, key="sw_sal_start")
                sweep_end = st.slider("Hệ số kết thúc (scale factor)", 0.80, 1.20, 1.20, step=0.05, key="sw_sal_end")
                sweep_step = st.slider("Bước nhảy (Step)", 0.01, 0.10, 0.05, step=0.01, key="sw_sal_step")
                
            run_sweep = st.button("Chạy Phân Tích Độ Nhạy 🚀", use_container_width=True, key="run_sweep_btn")
            
            if run_sweep:
                with st.spinner("Đang chạy phân tích độ nhạy..."):
                    if sweep_step <= 0:
                        st.error("Bước nhảy phải lớn hơn 0!")
                        values = []
                    else:
                        values = np.arange(sweep_start, sweep_end + sweep_step * 0.1, sweep_step).tolist()
                        
                    sweep_results = []
                    
                    # Read parameters from current widgets state
                    base_budget = budget
                    base_headcount = num_employees
                    base_job_titles = job_titles if job_titles else None
                    base_job_title_limits = job_title_limits if job_title_limits else None
                    base_job_title_max_ratio = job_title_max_ratio if job_title_max_ratio else None
                    base_allowed_exp = allowed_experience_years
                    base_exp_grps = exp_grps.copy() if exp_grps else {}
                    base_allowed_edu = allowed_education_levels if allowed_education_levels else None
                    base_edu_levels = education_levels if education_levels else None
                    base_edu_ratio_min = education_ratio_min if education_ratio_min > 0.0 else None
                    base_edu_ratio_max = education_ratio_max if education_ratio_max < 1.0 else None
                    base_min_skill = min_skill_count if min_skill_count > 0 else None
                    base_skills_high = skills_high
                    base_min_cert = min_certifications if min_certifications > 0 else None
                    base_certs_high = certs_high
                    base_allowed_remote = allowed_remote_types if allowed_remote_types else None
                    base_remote_types = remote_types if remote_types else None
                    base_remote_ratio_min = remote_ratio_min if remote_ratio_min > 0.0 else None
                    base_remote_ratio_max = remote_ratio_max if remote_ratio_max < 1.0 else None
                    base_min_quality = min_avg_quality_score if min_avg_quality_score > 0.0 else None
                    
                    for val in values:
                        sweep_budget = base_budget
                        sweep_headcount = base_headcount
                        sweep_exp_grps = base_exp_grps.copy()
                        sweep_remote_ratio_min = base_remote_ratio_min
                        sweep_remote_types = base_remote_types
                        df_run = df.copy()
                        
                        if sweep_param == "Ngân sách (Budget)":
                            sweep_budget = float(val)
                        elif sweep_param == "Số lượng tuyển (Headcount)":
                            sweep_headcount = int(val)
                        elif sweep_param == "Tỷ lệ Senior tối thiểu":
                            if 'Senior' not in sweep_exp_grps:
                                sweep_exp_grps['Senior'] = {}
                            else:
                                sweep_exp_grps['Senior'] = dict(sweep_exp_grps['Senior'])
                            sweep_exp_grps['Senior']['min_ratio'] = float(val)
                        elif sweep_param == "Tỷ lệ Remote tối thiểu":
                            sweep_remote_ratio_min = float(val)
                            if not sweep_remote_types:
                                sweep_remote_types = ["Remote"]
                            elif "Remote" not in sweep_remote_types:
                                sweep_remote_types = list(sweep_remote_types) + ["Remote"]
                        else:  # Hệ số chi phí lương
                            df_run['recruitment_cost'] = df_run['recruitment_cost'] * float(val)
                            
                        run_params = RecruitmentConstraints(
                            budget=sweep_budget,
                            num_employees=sweep_headcount,
                            job_titles=base_job_titles,
                            job_title_limits=base_job_title_limits,
                            job_title_max_ratio=base_job_title_max_ratio,
                            allowed_experience_years=base_allowed_exp if base_allowed_exp else None,
                            experience_groups=sweep_exp_grps if sweep_exp_grps else None,
                            allowed_education_levels=base_allowed_edu,
                            education_levels=base_edu_levels,
                            education_ratio_min=base_edu_ratio_min,
                            education_ratio_max=base_edu_ratio_max,
                            min_skill_count=base_min_skill,
                            skills_high=base_skills_high,
                            min_certifications=base_min_cert,
                            certifications_high=base_certs_high,
                            allowed_remote_types=base_allowed_remote,
                            remote_types=sweep_remote_types,
                            remote_ratio_min=sweep_remote_ratio_min,
                            remote_ratio_max=base_remote_ratio_max,
                            min_avg_quality_score=base_min_quality
                        )
                        
                        run_params, err = validate_and_fix(run_params)
                        if err:
                            res_entry = {
                                "Value": val,
                                "Status": "Infeasible (Input error)",
                                "Total Quality": 0.0,
                                "Avg Quality": 0.0,
                                "Total Cost": 0.0,
                                "Headcount": 0,
                                "Job Title Breakdown": {}
                            }
                        else:
                            res_df = solve_recruitment_ilp_engine(df_run, run_params)
                            if res_df is not None:
                                total_q = res_df['quality_score'].sum()
                                avg_q = res_df['quality_score'].mean()
                                total_c = res_df['recruitment_cost'].sum()
                                actual_hc = len(res_df)
                                breakdown = res_df['job_title'].value_counts().to_dict()
                                
                                res_entry = {
                                    "Value": val,
                                    "Status": "Optimal",
                                    "Total Quality": total_q,
                                    "Avg Quality": avg_q,
                                    "Total Cost": total_c,
                                    "Headcount": actual_hc,
                                    "Job Title Breakdown": breakdown
                                }
                            else:
                                res_entry = {
                                    "Value": val,
                                    "Status": "Infeasible",
                                    "Total Quality": 0.0,
                                    "Avg Quality": 0.0,
                                    "Total Cost": 0.0,
                                    "Headcount": 0,
                                    "Job Title Breakdown": {}
                                }
                        sweep_results.append(res_entry)
                        
                    st.session_state.sweep_results = sweep_results
                    st.session_state.sweep_param = sweep_param
                    st.session_state.sweep_values = values
                    
        with col_sweep_charts:
            if "sweep_results" in st.session_state:
                sweep_results = st.session_state.sweep_results
                sweep_param = st.session_state.sweep_param
                values = st.session_state.sweep_values
                
                plot_data = []
                comp_data = []
                
                for entry in sweep_results:
                    is_opt = entry["Status"] == "Optimal"
                    val = entry["Value"]
                    
                    plot_data.append({
                        "Value": val,
                        "Status": entry["Status"],
                        "Total Quality": entry["Total Quality"] if is_opt else None,
                        "Avg Quality": entry["Avg Quality"] if is_opt else None,
                        "Total Cost": entry["Total Cost"] if is_opt else None,
                        "Headcount": entry["Headcount"] if is_opt else None,
                    })
                    
                    all_possible_titles = ["AI Engineer", "Data Scientist", "Data Analyst", "Backend Developer", "Frontend Developer", "Machine Learning Engineer"]
                    breakdown = entry["Job Title Breakdown"]
                    for title in all_possible_titles:
                        comp_data.append({
                            "Value": val,
                            "Chức danh": title,
                            "Số lượng": breakdown.get(title, 0) if is_opt else 0
                        })
                        
                df_plot = pd.DataFrame(plot_data)
                df_comp = pd.DataFrame(comp_data)
                
                st.markdown(f"#### 📊 Kết Quả Phân Tích Độ Nhạy Theo: {sweep_param}")
                df_opt = df_plot[df_plot["Status"] == "Optimal"]
                
                if len(df_opt) == 0:
                    st.warning("⚠️ Không tìm thấy kịch bản khả thi (Optimal) nào trong phạm vi khảo sát. Vui lòng nới lỏng các ràng buộc cố định.")
                else:
                    # 1. Quality vs Parameter
                    fig_q = go.Figure()
                    fig_q.add_trace(go.Scatter(x=df_plot["Value"], y=df_plot["Total Quality"], mode='lines+markers', name='Tổng Chất Lượng', line=dict(color='#6366f1', width=3)))
                    fig_q.add_trace(go.Scatter(x=df_plot["Value"], y=df_plot["Avg Quality"] * 10, mode='lines+markers', name='Chất Lượng TB (*10)', line=dict(color='#ec4899', width=2, dash='dash')))
                    fig_q.update_layout(
                        title=f"Tương quan {sweep_param} vs. Chất lượng Đội ngũ",
                        xaxis_title=sweep_param,
                        yaxis_title="Điểm Chất Lượng",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color="#94a3b8"
                    )
                    st.plotly_chart(fig_q, use_container_width=True, key="sweep_quality")
                    
                    # 2. Headcount & Cost vs Parameter
                    fig_hc = go.Figure()
                    fig_hc.add_trace(go.Scatter(x=df_plot["Value"], y=df_plot["Headcount"], mode='lines+markers', name='Số lượng tuyển thực tế', line=dict(color='#10b981', width=3)))
                    fig_hc.add_trace(go.Scatter(x=df_plot["Value"], y=df_plot["Total Cost"] / 100000, mode='lines+markers', name='Tổng Chi Phí (x100k USD)', line=dict(color='#3b82f6', width=2)))
                    fig_hc.update_layout(
                        title=f"Tương quan {sweep_param} vs. Nhân sự & Chi phí",
                        xaxis_title=sweep_param,
                        yaxis_title="Số lượng / Chi phí",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color="#94a3b8"
                    )
                    st.plotly_chart(fig_hc, use_container_width=True, key="sweep_headcount")
                    
                    # 3. Job Title Composition Stacked Bar
                    fig_comp = px.bar(
                        df_comp, 
                        x="Value", 
                        y="Số lượng", 
                        color="Chức danh", 
                        title=f"Sự thay đổi cơ cấu chức danh theo {sweep_param}",
                        color_discrete_sequence=['#6366f1', '#a855f7', '#ec4899', '#3b82f6', '#10b981', '#f59e0b']
                    )
                    fig_comp.update_layout(
                        xaxis_title=sweep_param,
                        yaxis_title="Số lượng nhân sự",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color="#94a3b8"
                    )
                    st.plotly_chart(fig_comp, use_container_width=True, key="sweep_composition")
                    

                # 4. Summary Table
                st.markdown("##### 📋 Bảng tổng hợp kịch bản khảo sát")
                st.dataframe(df_plot, use_container_width=True)
                
                # Báo cáo độ nhạy tự động
                st.markdown("---")
                st.markdown("#### 📄 Báo Cáo Phân Tích Độ Nhạy Tự Động")
                
                def generate_sensitivity_report(sweep_results, sweep_param):
                    if not sweep_results:
                        return "Chưa có dữ liệu khảo sát độ nhạy."
                    
                    opt_results = [r for r in sweep_results if r["Status"] == "Optimal"]
                    if not opt_results:
                        return "Không có kịch bản tối ưu khả thi nào được tìm thấy trong khảo sát độ nhạy để phân tích."
                    
                    first = opt_results[0]
                    last = opt_results[-1]
                    
                    report = (
                        f"=== BÁO CÁO PHÂN TÍCH ĐỘ NHẠY DSS ===\n"
                        f"Tham số khảo sát: {sweep_param}\n"
                        f"Số kịch bản chạy thử: {len(sweep_results)} ({len(opt_results)} khả thi, {len(sweep_results)-len(opt_results)} không khả thi)\n\n"
                        f"--- Các phát hiện chính (Key Observations) ---\n"
                    )
                    
                    if sweep_param == "Ngân sách (Budget)":
                        report += (
                            f"- Khi ngân sách tăng từ ${first['Value']:,.2f} lên ${last['Value']:,.2f}:\n"
                            f"  * Tổng điểm chất lượng đội ngũ tăng từ {first['Total Quality']:.2f} lên {last['Total Quality']:.2f} (tăng {((last['Total Quality']-first['Total Quality'])/first['Total Quality']*100):.1f}%).\n"
                            f"  * Điểm chất lượng trung bình thay đổi từ {first['Avg Quality']:.2f} sang {last['Avg Quality']:.2f}.\n"
                            f"  * Tổng chi phí thực tế thay đổi từ ${first['Total Cost']:,.2f} sang ${last['Total Cost']:,.2f}.\n"
                        )
                    elif sweep_param == "Số lượng tuyển (Headcount)":
                        report += (
                            f"- Khi số lượng cần tuyển tăng từ {first['Value']} lên {last['Value']} người:\n"
                            f"  * Tổng chi phí thực tế tăng từ ${first['Total Cost']:,.2f} sang ${last['Total Cost']:,.2f}.\n"
                            f"  * Điểm chất lượng trung bình thay đổi từ {first['Avg Quality']:.2f} sang {last['Avg Quality']:.2f}.\n"
                        )
                    elif sweep_param == "Tỷ lệ Senior tối thiểu":
                        report += (
                            f"- Khi yêu cầu tỷ lệ Senior tối thiểu siết chặt từ {first['Value']*100:.0f}% lên {last['Value']*100:.0f}%:\n"
                            f"  * Tổng chi phí thực tế tăng từ ${first['Total Cost']:,.2f} sang ${last['Total Cost']:,.2f}.\n"
                            f"  * Điểm chất lượng trung bình thay đổi từ {first['Avg Quality']:.2f} sang {last['Avg Quality']:.2f}.\n"
                        )
                    elif sweep_param == "Tỷ lệ Remote tối thiểu":
                        report += (
                            f"- Khi yêu cầu tỷ lệ Remote tối thiểu thay đổi từ {first['Value']*100:.0f}% sang {last['Value']*100:.0f}%:\n"
                            f"  * Tổng chi phí thực tế tăng từ ${first['Total Cost']:,.2f} sang ${last['Total Cost']:,.2f}.\n"
                            f"  * Điểm chất lượng trung bình thay đổi từ {first['Avg Quality']:.2f} sang {last['Avg Quality']:.2f}.\n"
                        )
                    else:  # Chi phí lương
                        report += (
                            f"- Khi hệ số chi phí lương tăng từ {first['Value']*100:.0f}% lên {last['Value']*100:.0f}%:\n"
                            f"  * Tổng chi phí tuyển dụng thay đổi từ ${first['Total Cost']:,.2f} sang ${last['Total Cost']:,.2f}.\n"
                            f"  * Điểm chất lượng trung bình thay đổi từ {first['Avg Quality']:.2f} sang {last['Avg Quality']:.2f}.\n"
                        )
                        
                    infeasibles = [r for r in sweep_results if r["Status"] == "Infeasible"]
                    if infeasibles:
                        report += f"\n- Chú ý: Có {len(infeasibles)} kịch bản không khả thi (Infeasible) ở các mức giá trị: "
                        report += ", ".join([f"{r['Value']}" for r in infeasibles])
                        report += ". Doanh nghiệp nên tránh các thiết lập này vì chi phí thực tế vượt quá ngân sách giới hạn."
                        
                    report += "\n\n=== HẾT BÁO CÁO ==="
                    return report
                
                report_text = generate_sensitivity_report(sweep_results, sweep_param)
                st.text_area("Nội dung báo cáo độ nhạy", report_text, height=180, key="sens_report_textarea")
                
                rep_col1, rep_col2 = st.columns(2)
                with rep_col1:
                    st.download_button("Tải Báo Cáo Phân Tích Độ Nhạy (.txt) 📥", report_text, file_name="Sensitivity_Report.txt", key="download_sens_report")
                with rep_col2:
                    df_csv = pd.DataFrame(sweep_results)
                    csv_buffer = df_csv.to_csv(index=False).encode('utf-8')
                    st.download_button("Tải Dữ Liệu Khảo Sát (.csv) 📥", csv_buffer, "Sensitivity_Data.csv", "text/csv", key="download_sens_csv")
            else:
                st.info("💡 Thiết lập cấu hình bên trái và click **Chạy Phân Tích Độ Nhạy** để vẽ biểu đồ tác động.")
                
    with tabs[2]:
        st.markdown("### 💡 Phân Tích Giả Định (What-if Scenario Planning) & Báo Cáo")
        st.write(
            "So sánh đồng thời các kịch bản nhân sự khác nhau để đưa ra quyết định tối ưu "
            "về mặt phân bổ ngân sách và cơ cấu đội ngũ."
        )
        
        scenario_mode = st.selectbox(
            "Chọn kịch bản phân tích:",
            [
                "So sánh Tự do (Custom Comparison)", 
                "1. Kịch bản Lạc quan - Cơ sở - Bi quan (Market Fluctuations)", 
                "2. Kịch bản Mở rộng nhanh (Aggressive Growth)", 
                "3. Kịch bản Cắt giảm chi phí (Cost Reduction)"
            ],
            key="scenario_mode"
        )
        
        # Safe inheritance of variables
        base_budget = locals().get('budget', 1500000.0)
        base_headcount = locals().get('num_employees', 10)
        base_job_titles = locals().get('job_titles', None)
        base_job_title_limits = locals().get('job_title_limits', None)
        base_job_title_max_ratio = locals().get('job_title_max_ratio', None)
        base_allowed_exp = locals().get('allowed_experience_years', list(range(21)))
        base_exp_grps = locals().get('exp_grps', {}).copy()
        base_allowed_edu = locals().get('allowed_education_levels', None)
        base_edu_levels = locals().get('education_levels', None)
        base_edu_ratio_min = locals().get('education_ratio_min', None)
        base_edu_ratio_max = locals().get('education_ratio_max', None)
        base_min_skill = locals().get('min_skill_count', None)
        base_skills_high = locals().get('skills_high', None)
        base_min_cert = locals().get('min_certifications', None)
        base_certs_high = locals().get('certs_high', None)
        base_allowed_remote = locals().get('allowed_remote_types', None)
        base_remote_types = locals().get('remote_types', None)
        base_remote_ratio_min = locals().get('remote_ratio_min', None)
        base_remote_ratio_max = locals().get('remote_ratio_max', None)
        base_min_quality = locals().get('min_avg_quality_score', None)

        if scenario_mode == "So sánh Tự do (Custom Comparison)":
            col_scen_a, col_scen_b = st.columns(2)
            with col_scen_a:
                st.markdown("#### 💡 Kịch Bản A")
                scen_a_budget = st.number_input("Ngân sách A (USD)", min_value=10000.0, value=float(base_budget) * 1.2, step=50000.0, key="scen_a_budget")
                scen_a_hc = st.number_input("Số lượng tuyển A (người)", min_value=1, value=int(base_headcount), step=1, key="scen_a_hc")
                scen_a_senior = st.slider("Tỷ lệ Senior A", 0.0, 1.0, 0.30, step=0.05, key="scen_a_senior")
                scen_a_remote = st.slider("Tỷ lệ Remote A", 0.0, 1.0, 0.50, step=0.05, key="scen_a_remote")
                
            with col_scen_b:
                st.markdown("#### 💡 Kịch Bản B")
                scen_b_budget = st.number_input("Ngân sách B (USD)", min_value=10000.0, value=float(base_budget) * 0.9, step=50000.0, key="scen_b_budget")
                scen_b_hc = st.number_input("Số lượng tuyển B (người)", min_value=1, value=max(1, int(base_headcount) - 2), step=1, key="scen_b_hc")
                scen_b_senior = st.slider("Tỷ lệ Senior B", 0.0, 1.0, 0.20, step=0.05, key="scen_b_senior")
                scen_b_remote = st.slider("Tỷ lệ Remote B", 0.0, 1.0, 0.70, step=0.05, key="scen_b_remote")
                
            run_compare = st.button("So Sánh Kịch Bản 💡", use_container_width=True, key="run_compare")
            
            if run_compare:
                with st.spinner("Đang tính toán các kịch bản..."):
                    compare_results = []
                    scenarios = [
                        {"Name": "Kịch Bản Hiện Tại", "Budget": base_budget, "Headcount": base_headcount, "Senior": base_exp_grps.get('Senior', {}).get('min_ratio', 0.0), "Remote": base_remote_ratio_min, "CostScale": 1.0},
                        {"Name": "Kịch Bản A", "Budget": scen_a_budget, "Headcount": scen_a_hc, "Senior": scen_a_senior, "Remote": scen_a_remote, "CostScale": 1.0},
                        {"Name": "Kịch Bản B", "Budget": scen_b_budget, "Headcount": scen_b_hc, "Senior": scen_b_senior, "Remote": scen_b_remote, "CostScale": 1.0}
                    ]
                    
                    for s in scenarios:
                        s_exp_grps = base_exp_grps.copy()
                        if s["Senior"] > 0:
                            if 'Senior' not in s_exp_grps:
                                s_exp_grps['Senior'] = {}
                            else:
                                s_exp_grps['Senior'] = dict(s_exp_grps['Senior'])
                            s_exp_grps['Senior']['min_ratio'] = s["Senior"]
                            
                        s_remote_types = base_remote_types
                        s_remote_ratio_min = s["Remote"]
                        if s_remote_ratio_min and s_remote_ratio_min > 0.0:
                            if not s_remote_types:
                                s_remote_types = ["Remote"]
                            elif "Remote" not in s_remote_types:
                                s_remote_types = list(s_remote_types) + ["Remote"]
                                
                        run_params = RecruitmentConstraints(
                            budget=s["Budget"],
                            num_employees=s["Headcount"],
                            job_titles=base_job_titles,
                            job_title_limits=base_job_title_limits,
                            job_title_max_ratio=base_job_title_max_ratio,
                            allowed_experience_years=base_allowed_exp if base_allowed_exp else None,
                            experience_groups=s_exp_grps if s_exp_grps else None,
                            allowed_education_levels=base_allowed_edu,
                            education_levels=base_edu_levels,
                            education_ratio_min=base_edu_ratio_min,
                            education_ratio_max=base_edu_ratio_max,
                            min_skill_count=base_min_skill,
                            skills_high=base_skills_high,
                            min_certifications=base_min_cert,
                            certifications_high=base_certs_high,
                            allowed_remote_types=base_allowed_remote,
                            remote_types=s_remote_types,
                            remote_ratio_min=s_remote_ratio_min,
                            remote_ratio_max=base_remote_ratio_max,
                            min_avg_quality_score=base_min_quality
                        )
                        
                        run_params, err = validate_and_fix(run_params)
                        if err:
                            compare_results.append({"Kịch bản": s["Name"], "Trạng thái": "Infeasible (Input error)", "Ngân sách tối đa": s["Budget"], "Số lượng cần tuyển": s["Headcount"], "Số lượng tuyển thực tế": 0, "Tổng chi phí tuyển dụng": 0.0, "Điểm chất lượng trung bình": 0.0, "Tổng điểm chất lượng": 0.0, "Hiệu suất sử dụng ngân sách": 0.0})
                        else:
                            res_df = solve_recruitment_ilp_engine(df, run_params)
                            if res_df is not None:
                                cost_sum = res_df['recruitment_cost'].sum()
                                compare_results.append({
                                    "Kịch bản": s["Name"],
                                    "Trạng thái": "Optimal",
                                    "Ngân sách tối đa": s["Budget"],
                                    "Số lượng cần tuyển": s["Headcount"],
                                    "Số lượng tuyển thực tế": len(res_df),
                                    "Tổng chi phí tuyển dụng": cost_sum,
                                    "Điểm chất lượng trung bình": res_df['quality_score'].mean(),
                                    "Tổng điểm chất lượng": res_df['quality_score'].sum(),
                                    "Hiệu suất sử dụng ngân sách": (cost_sum / s["Budget"]) * 100,
                                    "Breakdown": res_df['job_title'].value_counts().to_dict()
                                })
                            else:
                                compare_results.append({"Kịch bản": s["Name"], "Trạng thái": "Infeasible", "Ngân sách tối đa": s["Budget"], "Số lượng cần tuyển": s["Headcount"], "Số lượng tuyển thực tế": 0, "Tổng chi phí tuyển dụng": 0.0, "Điểm chất lượng trung bình": 0.0, "Tổng điểm chất lượng": 0.0, "Hiệu suất sử dụng ngân sách": 0.0})
                                
                    st.session_state.compare_results = compare_results
                    st.session_state.compare_mode = "Custom"

        elif scenario_mode == "1. Kịch bản Lạc quan - Cơ sở - Bi quan (Market Fluctuations)":
            st.markdown("#### 📊 Bối cảnh Phân Tích")
            st.write(
                "Đánh giá tác động của biến động chi phí nhân sự trên thị trường lao động (lương tăng/giảm) "
                "và sự thay đổi tương ứng về ngân sách phân bổ của doanh nghiệp."
            )
            
            st.markdown(f"""
            | Kịch bản | Biến động Chi phí nhân sự | Biến động Ngân sách | Ngân sách cụ thể |
            | :--- | :--- | :--- | :--- |
            | **Lạc quan (Optimistic)** | -10% | +10% | **${base_budget * 1.1:,.2f}** |
            | **Cơ sở (Base)** | 0% | 0% | **${base_budget:,.2f}** |
            | **Bi quan (Pessimistic)** | +20% | -10% | **${base_budget * 0.9:,.2f}** |
            """)
            
            run_macro = st.button("Chạy Phân Tích Lạc Quan - Cơ Sở - Bi Quan 🚀", use_container_width=True, key="run_macro")
            
            if run_macro:
                with st.spinner("Đang tính toán các kịch bản thị trường..."):
                    compare_results = []
                    scenarios = [
                        {"Name": "Lạc Quan (Optimistic)", "Budget": base_budget * 1.1, "Headcount": base_headcount, "CostScale": 0.9},
                        {"Name": "Cơ Sở (Base)", "Budget": base_budget, "Headcount": base_headcount, "CostScale": 1.0},
                        {"Name": "Bi Quan (Pessimistic)", "Budget": base_budget * 0.9, "Headcount": base_headcount, "CostScale": 1.2}
                    ]
                    
                    for s in scenarios:
                        df_run = df.copy()
                        df_run['recruitment_cost'] = df_run['recruitment_cost'] * s["CostScale"]
                        
                        run_params = RecruitmentConstraints(
                            budget=s["Budget"],
                            num_employees=s["Headcount"],
                            job_titles=base_job_titles,
                            job_title_limits=base_job_title_limits,
                            job_title_max_ratio=base_job_title_max_ratio,
                            allowed_experience_years=base_allowed_exp if base_allowed_exp else None,
                            experience_groups=base_exp_grps if base_exp_grps else None,
                            allowed_education_levels=base_allowed_edu,
                            education_levels=base_edu_levels,
                            education_ratio_min=base_edu_ratio_min,
                            education_ratio_max=base_edu_ratio_max,
                            min_skill_count=base_min_skill,
                            skills_high=base_skills_high,
                            min_certifications=base_min_cert,
                            certifications_high=base_certs_high,
                            allowed_remote_types=base_allowed_remote,
                            remote_types=base_remote_types,
                            remote_ratio_min=base_remote_ratio_min,
                            remote_ratio_max=base_remote_ratio_max,
                            min_avg_quality_score=base_min_quality
                        )
                        
                        run_params, err = validate_and_fix(run_params)
                        if err:
                            compare_results.append({"Kịch bản": s["Name"], "Trạng thái": "Infeasible (Input error)", "Ngân sách tối đa": s["Budget"], "Số lượng cần tuyển": s["Headcount"], "Số lượng tuyển thực tế": 0, "Tổng chi phí tuyển dụng": 0.0, "Điểm chất lượng trung bình": 0.0, "Tổng điểm chất lượng": 0.0, "Hiệu suất sử dụng ngân sách": 0.0})
                        else:
                            res_df = solve_recruitment_ilp_engine(df_run, run_params)
                            if res_df is not None:
                                cost_sum = res_df['recruitment_cost'].sum()
                                compare_results.append({
                                    "Kịch bản": s["Name"],
                                    "Trạng thái": "Optimal",
                                    "Ngân sách tối đa": s["Budget"],
                                    "Số lượng cần tuyển": s["Headcount"],
                                    "Số lượng tuyển thực tế": len(res_df),
                                    "Tổng chi phí tuyển dụng": cost_sum,
                                    "Điểm chất lượng trung bình": res_df['quality_score'].mean(),
                                    "Tổng điểm chất lượng": res_df['quality_score'].sum(),
                                    "Hiệu suất sử dụng ngân sách": (cost_sum / s["Budget"]) * 100,
                                    "Breakdown": res_df['job_title'].value_counts().to_dict()
                                })
                            else:
                                compare_results.append({"Kịch bản": s["Name"], "Trạng thái": "Infeasible", "Ngân sách tối đa": s["Budget"], "Số lượng cần tuyển": s["Headcount"], "Số lượng tuyển thực tế": 0, "Tổng chi phí tuyển dụng": 0.0, "Điểm chất lượng trung bình": 0.0, "Tổng điểm chất lượng": 0.0, "Hiệu suất sử dụng ngân sách": 0.0})
                                
                    st.session_state.compare_results = compare_results
                    st.session_state.compare_mode = "Market"

        elif scenario_mode == "2. Kịch bản Mở rộng nhanh (Aggressive Growth)":
            st.markdown("#### 🚀 Bối cảnh Phân Tích")
            st.write(
                "Doanh nghiệp nhận được vốn đầu tư mới, cần mở rộng nhanh đội ngũ kỹ sư Data/AI "
                "với yêu cầu cao hơn về mặt chất lượng (ít nhất 30% Senior)."
            )
            
            st.markdown(f"""
            | Tham số | Kịch Bản Cơ Sở | Kịch Bản Mở Rộng Nhanh |
            | :--- | :--- | :--- |
            | **Ngân sách** | ${base_budget:,.2f} | **+50%** (${base_budget * 1.5:,.2f}) |
            | **Số lượng tuyển** | {base_headcount} người | **+70%** ({int(base_headcount * 1.7)} người) |
            | **Tỷ lệ Senior tối thiểu** | {base_exp_grps.get('Senior', {}).get('min_ratio', 0.0)*100:.0f}% | **30%** |
            """)
            
            run_growth = st.button("Chạy Phân Tích Mở Rộng Nhanh 🚀", use_container_width=True, key="run_growth")
            
            if run_growth:
                with st.spinner("Đang tính toán kịch bản mở rộng nhanh..."):
                    compare_results = []
                    
                    # 1. Base solver
                    run_params_base = RecruitmentConstraints(
                        budget=base_budget,
                        num_employees=base_headcount,
                        job_titles=base_job_titles,
                        job_title_limits=base_job_title_limits,
                        job_title_max_ratio=base_job_title_max_ratio,
                        allowed_experience_years=base_allowed_exp if base_allowed_exp else None,
                        experience_groups=base_exp_grps if base_exp_grps else None,
                        allowed_education_levels=base_allowed_edu,
                        education_levels=base_edu_levels,
                        education_ratio_min=base_edu_ratio_min,
                        education_ratio_max=base_edu_ratio_max,
                        min_skill_count=base_min_skill,
                        skills_high=base_skills_high,
                        min_certifications=base_min_cert,
                        certifications_high=base_certs_high,
                        allowed_remote_types=base_allowed_remote,
                        remote_types=base_remote_types,
                        remote_ratio_min=base_remote_ratio_min,
                        remote_ratio_max=base_remote_ratio_max,
                        min_avg_quality_score=base_min_quality
                    )
                    
                    # 2. Aggressive Growth solver
                    growth_exp_grps = base_exp_grps.copy()
                    if 'Senior' not in growth_exp_grps:
                        growth_exp_grps['Senior'] = {}
                    else:
                        growth_exp_grps['Senior'] = dict(growth_exp_grps['Senior'])
                    growth_exp_grps['Senior']['min_ratio'] = 0.30
                    
                    run_params_growth = RecruitmentConstraints(
                        budget=base_budget * 1.5,
                        num_employees=int(base_headcount * 1.7),
                        job_titles=base_job_titles,
                        job_title_limits=base_job_title_limits,
                        job_title_max_ratio=base_job_title_max_ratio,
                        allowed_experience_years=base_allowed_exp if base_allowed_exp else None,
                        experience_groups=growth_exp_grps,
                        allowed_education_levels=base_allowed_edu,
                        education_levels=base_edu_levels,
                        education_ratio_min=base_edu_ratio_min,
                        education_ratio_max=base_edu_ratio_max,
                        min_skill_count=base_min_skill,
                        skills_high=base_skills_high,
                        min_certifications=base_min_cert,
                        certifications_high=base_certs_high,
                        allowed_remote_types=base_allowed_remote,
                        remote_types=base_remote_types,
                        remote_ratio_min=base_remote_ratio_min,
                        remote_ratio_max=base_remote_ratio_max,
                        min_avg_quality_score=base_min_quality
                    )
                    
                    # Solve Base
                    base_df = solve_recruitment_ilp_engine(df, run_params_base)
                    if base_df is not None:
                        cost_sum = base_df['recruitment_cost'].sum()
                        compare_results.append({
                            "Kịch bản": "Cơ Sở (Base)",
                            "Trạng thái": "Optimal",
                            "Ngân sách tối đa": base_budget,
                            "Số lượng cần tuyển": base_headcount,
                            "Số lượng tuyển thực tế": len(base_df),
                            "Tổng chi phí tuyển dụng": cost_sum,
                            "Điểm chất lượng trung bình": base_df['quality_score'].mean(),
                            "Tổng điểm chất lượng": base_df['quality_score'].sum(),
                            "Hiệu suất sử dụng ngân sách": (cost_sum / base_budget) * 100,
                            "Breakdown": base_df['job_title'].value_counts().to_dict()
                        })
                    else:
                        compare_results.append({"Kịch bản": "Cơ Sở (Base)", "Trạng thái": "Infeasible", "Ngân sách tối đa": base_budget, "Số lượng cần tuyển": base_headcount, "Số lượng tuyển thực tế": 0, "Tổng chi phí tuyển dụng": 0.0, "Điểm chất lượng trung bình": 0.0, "Tổng điểm chất lượng": 0.0, "Hiệu suất sử dụng ngân sách": 0.0, "Breakdown": {}})
                        
                    # Solve Growth
                    growth_df = solve_recruitment_ilp_engine(df, run_params_growth)
                    if growth_df is not None:
                        cost_sum = growth_df['recruitment_cost'].sum()
                        compare_results.append({
                            "Kịch bản": "Mở Rộng Nhanh (Aggressive Growth)",
                            "Trạng thái": "Optimal",
                            "Ngân sách tối đa": base_budget * 1.5,
                            "Số lượng cần tuyển": int(base_headcount * 1.7),
                            "Số lượng tuyển thực tế": len(growth_df),
                            "Tổng chi phí tuyển dụng": cost_sum,
                            "Điểm chất lượng trung bình": growth_df['quality_score'].mean(),
                            "Tổng điểm chất lượng": growth_df['quality_score'].sum(),
                            "Hiệu suất sử dụng ngân sách": (cost_sum / (base_budget * 1.5)) * 100,
                            "Breakdown": growth_df['job_title'].value_counts().to_dict()
                        })
                    else:
                        compare_results.append({"Kịch bản": "Mở Rộng Nhanh (Aggressive Growth)", "Trạng thái": "Infeasible", "Ngân sách tối đa": base_budget * 1.5, "Số lượng cần tuyển": int(base_headcount * 1.7), "Số lượng tuyển thực tế": 0, "Tổng chi phí tuyển dụng": 0.0, "Điểm chất lượng trung bình": 0.0, "Tổng điểm chất lượng": 0.0, "Hiệu suất sử dụng ngân sách": 0.0, "Breakdown": {}})
                        
                    st.session_state.compare_results = compare_results
                    st.session_state.compare_mode = "Growth"

        else:  # scenario_mode == "3. Kịch bản Cắt giảm chi phí (Cost Reduction)"
            st.markdown("#### 📉 Bối cảnh Phân Tích")
            st.write(
                "Doanh nghiệp buộc phải cắt giảm ngân sách tuyển dụng do suy thoái kinh tế "
                "nhưng vẫn muốn duy trì tối đa chất lượng đội ngũ (giữ nguyên yêu cầu chất lượng tối thiểu)."
            )
            
            st.markdown(f"""
            | Tham số | Kịch Bản Cơ Sở | Kịch Bản Cắt Giảm Chi Phí |
            | :--- | :--- | :--- |
            | **Ngân sách** | ${base_budget:,.2f} | **-30%** (${base_budget * 0.7:,.2f}) |
            | **Số lượng tuyển** | {base_headcount} người | {base_headcount} người |
            | **Điểm chất lượng tối thiểu** | {base_min_quality if base_min_quality else 'Không yêu cầu'} | {base_min_quality if base_min_quality else 'Không yêu cầu'} |
            """)
            
            run_cut = st.button("Chạy Phân Tích Cắt Giảm Chi Phí 🚀", use_container_width=True, key="run_cut")
            
            if run_cut:
                with st.spinner("Đang tính toán kịch bản cắt giảm chi phí..."):
                    compare_results = []
                    
                    # 1. Base solver
                    run_params_base = RecruitmentConstraints(
                        budget=base_budget,
                        num_employees=base_headcount,
                        job_titles=base_job_titles,
                        job_title_limits=base_job_title_limits,
                        job_title_max_ratio=base_job_title_max_ratio,
                        allowed_experience_years=base_allowed_exp if base_allowed_exp else None,
                        experience_groups=base_exp_grps if base_exp_grps else None,
                        allowed_education_levels=base_allowed_edu,
                        education_levels=base_edu_levels,
                        education_ratio_min=base_edu_ratio_min,
                        education_ratio_max=base_edu_ratio_max,
                        min_skill_count=base_min_skill,
                        skills_high=base_skills_high,
                        min_certifications=base_min_cert,
                        certifications_high=base_certs_high,
                        allowed_remote_types=base_allowed_remote,
                        remote_types=base_remote_types,
                        remote_ratio_min=base_remote_ratio_min,
                        remote_ratio_max=base_remote_ratio_max,
                        min_avg_quality_score=base_min_quality
                    )
                    
                    # 2. Cost Reduction solver
                    run_params_cut = RecruitmentConstraints(
                        budget=base_budget * 0.7,
                        num_employees=base_headcount,
                        job_titles=base_job_titles,
                        job_title_limits=base_job_title_limits,
                        job_title_max_ratio=base_job_title_max_ratio,
                        allowed_experience_years=base_allowed_exp if base_allowed_exp else None,
                        experience_groups=base_exp_grps if base_exp_grps else None,
                        allowed_education_levels=base_allowed_edu,
                        education_levels=base_edu_levels,
                        education_ratio_min=base_edu_ratio_min,
                        education_ratio_max=base_edu_ratio_max,
                        min_skill_count=base_min_skill,
                        skills_high=base_skills_high,
                        min_certifications=base_min_cert,
                        certifications_high=base_certs_high,
                        allowed_remote_types=base_allowed_remote,
                        remote_types=base_remote_types,
                        remote_ratio_min=base_remote_ratio_min,
                        remote_ratio_max=base_remote_ratio_max,
                        min_avg_quality_score=base_min_quality
                    )
                    
                    # Solve Base
                    base_df = solve_recruitment_ilp_engine(df, run_params_base)
                    if base_df is not None:
                        cost_sum = base_df['recruitment_cost'].sum()
                        compare_results.append({
                            "Kịch bản": "Cơ Sở (Base)",
                            "Trạng thái": "Optimal",
                            "Ngân sách tối đa": base_budget,
                            "Số lượng cần tuyển": base_headcount,
                            "Số lượng tuyển thực tế": len(base_df),
                            "Tổng chi phí tuyển dụng": cost_sum,
                            "Điểm chất lượng trung bình": base_df['quality_score'].mean(),
                            "Tổng điểm chất lượng": base_df['quality_score'].sum(),
                            "Hiệu suất sử dụng ngân sách": (cost_sum / base_budget) * 100,
                            "Breakdown": base_df['job_title'].value_counts().to_dict()
                        })
                    else:
                        compare_results.append({"Kịch bản": "Cơ Sở (Base)", "Trạng thái": "Infeasible", "Ngân sách tối đa": base_budget, "Số lượng cần tuyển": base_headcount, "Số lượng tuyển thực tế": 0, "Tổng chi phí tuyển dụng": 0.0, "Điểm chất lượng trung bình": 0.0, "Tổng điểm chất lượng": 0.0, "Hiệu suất sử dụng ngân sách": 0.0, "Breakdown": {}})
                        
                    # Solve Cut
                    cut_df = solve_recruitment_ilp_engine(df, run_params_cut)
                    if cut_df is not None:
                        cost_sum = cut_df['recruitment_cost'].sum()
                        compare_results.append({
                            "Kịch bản": "Cắt Giảm Chi Phí (Cost Reduction)",
                            "Trạng thái": "Optimal",
                            "Ngân sách tối đa": base_budget * 0.7,
                            "Số lượng cần tuyển": base_headcount,
                            "Số lượng tuyển thực tế": len(cut_df),
                            "Tổng chi phí tuyển dụng": cost_sum,
                            "Điểm chất lượng trung bình": cut_df['quality_score'].mean(),
                            "Tổng điểm chất lượng": cut_df['quality_score'].sum(),
                            "Hiệu suất sử dụng ngân sách": (cost_sum / (base_budget * 0.7)) * 100,
                            "Breakdown": cut_df['job_title'].value_counts().to_dict()
                        })
                    else:
                        compare_results.append({"Kịch bản": "Cắt Giảm Chi Phí (Cost Reduction)", "Trạng thái": "Infeasible", "Ngân sách tối đa": base_budget * 0.7, "Số lượng cần tuyển": base_headcount, "Số lượng tuyển thực tế": 0, "Tổng chi phí tuyển dụng": 0.0, "Điểm chất lượng trung bình": 0.0, "Tổng điểm chất lượng": 0.0, "Hiệu suất sử dụng ngân sách": 0.0, "Breakdown": {}})
                        
                    st.session_state.compare_results = compare_results
                    st.session_state.compare_mode = "Cut"

        # Show comparison results if available
        if "compare_results" in st.session_state and st.session_state.compare_results:
            compare_results = st.session_state.compare_results
            compare_mode = st.session_state.get("compare_mode", "Custom")
            
            st.markdown("#### 📊 Kết Quả So Sánh")
            df_compare = pd.DataFrame(compare_results)
            df_show = df_compare.drop("Breakdown", axis=1) if "Breakdown" in df_compare.columns else df_compare.copy()
            st.dataframe(df_show, use_container_width=True)
            
            # Display side-by-side metric cards
            sc_cols = st.columns(len(compare_results))
            for idx, item in enumerate(compare_results):
                with sc_cols[idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">{item["Kịch bản"]}</div>
                        <div style="font-size: 1.2rem; font-weight: bold; margin: 5px 0; color: {'#10b981' if item['Trạng thái'] == 'Optimal' else '#ef4444'}">Trạng thái: {item["Trạng thái"]}</div>
                        <div>Tuyển thực tế: <b>{item["Số lượng tuyển thực tế"]} người</b></div>
                        <div>Tổng chi phí: <b>${item["Tổng chi phí tuyển dụng"]:,.2f}</b></div>
                        <div>Chất lượng TB: <b>{item["Điểm chất lượng trung bình"]:.2f}/10</b></div>
                        <div>Sử dụng ngân sách: <b>{item["Hiệu suất sử dụng ngân sách"]:.1f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            # Text Analysis & Báo cáo
            st.markdown("---")
            st.markdown("#### 📄 Báo Cáo Phân Tích DSS What-if")
            
            def generate_whatif_report(results, mode):
                if not results:
                    return "Chưa có dữ liệu so sánh kịch bản."
                
                report = f"=== BÁO CÁO PHÂN TÍCH WHAT-IF SCENARIOS ===\nChế độ phân tích: {mode}\n\n"
                opt_dict = {r["Kịch bản"]: r for r in results if r["Trạng thái"] == "Optimal"}
                
                if mode == "Market":
                    report += (
                        "=== PHÂN TÍCH KỊCH BẢN BIẾN ĐỘNG THỊ TRƯỜNG (LẠC QUAN - CƠ SỞ - BI QUAN) ===\n"
                        "Câu hỏi quản lý: Nếu thị trường AI trở nên cạnh tranh hơn và chi phí nhân sự tăng mạnh (+20% chi phí) và ngân sách giảm 10% (Bi quan), công ty có còn đạt được mục tiêu tuyển dụng không?\n\n"
                    )
                    
                    if "Bi Quan (Pessimistic)" in opt_dict:
                        item = opt_dict["Bi Quan (Pessimistic)"]
                        report += (
                            f"Kết quả phân tích: CÓ THỂ ĐẠT ĐƯỢC!\n"
                            f"- Trong điều kiện Bi quan, hệ thống vẫn giải bài toán ILP thành công.\n"
                            f"- Số lượng nhân sự tuyển được: {item['Số lượng tuyển thực tế']} / {item['Số lượng cần tuyển']} người.\n"
                            f"- Tổng chi phí tuyển dụng thực tế: ${item['Tổng chi phí tuyển dụng']:,.2f} (dưới ngân sách thắt chặt ${item['Ngân sách tối đa']:,.2f}).\n"
                            f"- Điểm chất lượng trung bình của đội ngũ: {item['Điểm chất lượng trung bình']:.2f}/10.\n"
                            f"- Kết luận: Công ty vẫn có khả năng hoàn thành chỉ tiêu tuyển dụng nhờ vào cơ sở dữ liệu ứng viên đa dạng, tối ưu hóa ILP giúp lựa chọn chính xác các ứng viên có chi phí tối ưu nhất nhưng vẫn đáp ứng tiêu chuẩn.\n"
                        )
                    else:
                        report += (
                            "Kết quả phân tích: KHÔNG THỂ ĐẠT ĐƯỢC!\n"
                            "- Trong kịch bản Bi quan, mô hình toán học báo KHÔNG KHẢ THI (Infeasible).\n"
                            f"- Lý do: Chi phí nhân sự thị trường tăng 20% và ngân sách bị cắt giảm 10% khiến không còn phương án phân bổ nào thỏa mãn đồng thời các tiêu chí.\n"
                            "- Khuyến nghị: Công ty nên hoãn chỉ tiêu tuyển dụng, giảm số lượng cần tuyển hoặc nới lỏng các ràng buộc về học vấn/hình thức làm việc/kinh nghiệm để giảm chi phí.\n"
                        )
                        
                    if "Lạc Quan (Optimistic)" in opt_dict:
                        item = opt_dict["Lạc Quan (Optimistic)"]
                        report += (
                            f"\nTrong điều kiện Lạc quan (Chi phí tuyển dụng -10%, Ngân sách +10%):\n"
                            f"- Đội ngũ tuyển dụng đạt chất lượng trung bình rất cao: {item['Điểm chất lượng trung bình']:.2f}/10.\n"
                            f"- Tổng chi phí thực tế là: ${item['Tổng chi phí tuyển dụng']:,.2f} (chỉ sử dụng {item['Hiệu suất sử dụng ngân sách']:.1f}% ngân sách).\n"
                        )
                        
                elif mode == "Growth":
                    report += (
                        "=== PHÂN TÍCH KỊCH BẢN MỞ RỘNG NHANH (AGGRESSIVE GROWTH) ===\n"
                        "Câu hỏi quản lý: Với tốc độ tăng trưởng cao (+50% ngân sách, +70% headcount, >=30% Senior), công ty nên phân bổ ngân sách tuyển dụng như thế nào?\n\n"
                    )
                    
                    if "Mở Rộng Nhanh (Aggressive Growth)" in opt_dict:
                        item = opt_dict["Mở Rộng Nhanh (Aggressive Growth)"]
                        report += (
                            "Kết quả phân tích: KỊCH BẢN KHẢ THI VÀ TỐI ƯU!\n"
                            f"- Số lượng nhân sự tuyển được: {item['Số lượng tuyển thực tế']} người.\n"
                            f"- Tổng chi phí tuyển dụng thực tế: ${item['Tổng chi phí tuyển dụng']:,.2f} (Sử dụng {item['Hiệu suất sử dụng ngân sách']:.1f}% trên tổng ngân sách mở rộng ${item['Ngân sách tối đa']:,.2f}).\n"
                            f"- Điểm chất lượng trung bình của đội ngũ: {item['Điểm chất lượng trung bình']:.2f}/10.\n"
                            f"- Phân bổ cơ cấu chức danh tối ưu đề xuất:\n"
                        )
                        breakdown = item.get("Breakdown", {})
                        for title, count in breakdown.items():
                            report += f"  * {title}: {count} người.\n"
                    else:
                        report += (
                            "Kết quả phân tích: KỊCH BẢN KHÔNG KHẢ THI (Infeasible)!\n"
                            "- Lý do: Nguồn cung ứng viên Senior trong database không đủ đáp ứng tỷ lệ 30% cho lượng headcount mở rộng lớn (+70%), hoặc ngân sách tăng 50% vẫn chưa đủ trang trải cho đội hình có nhiều Senior.\n"
                            "- Khuyến nghị: Doanh nghiệp nên hạ tỷ lệ Senior yêu cầu xuống 20% hoặc tăng ngân sách mở rộng thêm 20% nữa.\n"
                        )
                        
                elif mode == "Cut":
                    report += (
                        "=== PHÂN TÍCH KỊCH BẢN CẮT GIẢM CHI PHÍ (COST REDUCTION) ===\n"
                        "Câu hỏi quản lý: Khi ngân sách bị cắt giảm 30%, nên giảm tuyển ở vị trí nào để vẫn duy trì chất lượng đội ngũ?\n\n"
                    )
                    
                    if "Cắt Giảm Chi Phí (Cost Reduction)" in opt_dict:
                        item = opt_dict["Cắt Giảm Chi Phí (Cost Reduction)"]
                        base_item = opt_dict.get("Cơ Sở (Base)")
                        report += (
                            f"- Tổng chi phí thực tế: ${item['Tổng chi phí tuyển dụng']:,.2f}.\n"
                            f"- Điểm chất lượng trung bình: {item['Điểm chất lượng trung bình']:.2f}/10 (so với Cơ sở: {base_item['Điểm chất lượng trung bình']:.2f}/10, mức giảm chất lượng là {abs(item['Điểm chất lượng trung bình']-base_item['Điểm chất lượng trung bình']):.2f} điểm).\n\n"
                            "- Đánh giá thay đổi cơ cấu chức danh:\n"
                        )
                        
                        base_breakdown = base_item.get("Breakdown", {}) if base_item else {}
                        cut_breakdown = item.get("Breakdown", {})
                        all_titles = set(list(base_breakdown.keys()) + list(cut_breakdown.keys()))
                        
                        for title in all_titles:
                            b_cnt = base_breakdown.get(title, 0)
                            c_cnt = cut_breakdown.get(title, 0)
                            diff = c_cnt - b_cnt
                            if diff < 0:
                                report += f"  * {title}: {b_cnt} -> {c_cnt} người (Cắt giảm {-diff} vị trí - ĐÂY LÀ VỊ TRÍ BỊ GIẢM ĐẦU TIÊN)\n"
                            elif diff > 0:
                                report += f"  * {title}: {b_cnt} -> {c_cnt} người (Tăng {diff} vị trí có chi phí thấp hơn để bù đắp headcount)\n"
                            else:
                                report += f"  * {title}: Giữ nguyên {b_cnt} người.\n"
                    else:
                        report += (
                            "Kết quả phân tích: KỊCH BẢN KHÔNG KHẢ THI (Infeasible)!\n"
                            "- Lý do: Cắt giảm 30% ngân sách khiến công ty không thể tuyển đủ số lượng headcount và duy trì chất lượng tối thiểu.\n"
                            "- Khuyến nghị: Công ty buộc phải chấp nhận hạ thấp yêu cầu điểm chất lượng trung bình hoặc giảm chỉ tiêu số lượng cần tuyển xuống.\n"
                        )
                else:
                    report += "=== PHÂN TÍCH SO SÁNH TỰ DO ===\n"
                    for r in results:
                        report += f"- {r['Kịch bản']}: Trạng thái: {r['Trạng thái']}, Tuyển thực tế: {r['Số lượng tuyển thực tế']} người, Chi phí: ${r['Tổng chi phí tuyển dụng']:,.2f}, Chất lượng TB: {r['Điểm chất lượng trung bình']:.2f}/10\n"
                
                report += "\n=== HẾT BÁO CÁO ==="
                return report
                
            report_text = generate_whatif_report(compare_results, compare_mode)
            st.text_area("Nội dung báo cáo kịch bản", report_text, height=250, key="scen_report_textarea")
            
            # Download buttons
            rep_col1, rep_col2 = st.columns(2)
            with rep_col1:
                st.download_button("Tải Báo Cáo Kịch Bản (.txt) 📥", report_text, file_name="Scenario_Report.txt", key="download_scen_report")
            with rep_col2:
                df_csv = pd.DataFrame(compare_results)
                if "Breakdown" in df_csv.columns:
                    df_csv = df_csv.drop("Breakdown", axis=1)
                csv_buffer = df_csv.to_csv(index=False).encode('utf-8')
                st.download_button("Tải Dữ Liệu Kịch Bản (.csv) 📥", csv_buffer, "Scenario_Comparison_Data.csv", "text/csv", key="download_scen_csv")
                
            # Render visual comparison chart if appropriate
            if compare_mode in ["Growth", "Cut"] and len(compare_results) >= 2:
                opt_results = [r for r in compare_results if r["Trạng thái"] == "Optimal"]
                if len(opt_results) >= 2:
                    st.markdown("#### 📊 Đồ Thị So Sánh Cơ Cấu Chức Danh")
                    chart_data = []
                    for r in opt_results:
                        breakdown = r.get("Breakdown", {})
                        for title, count in breakdown.items():
                            chart_data.append({
                                "Kịch bản": r["Kịch bản"],
                                "Chức danh": title,
                                "Số lượng": count
                            })
                    df_chart = pd.DataFrame(chart_data)
                    fig_compare = px.bar(
                        df_chart, 
                        x="Chức danh", 
                        y="Số lượng", 
                        color="Kịch bản", 
                        barmode="group",
                        title="So sánh phân bổ chức danh tuyển dụng",
                        color_discrete_sequence=['#6366f1', '#ec4899']
                    )
                    fig_compare.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color="#94a3b8"
                    )
                    st.plotly_chart(fig_compare, use_container_width=True, key="whatif_compare")
            elif compare_mode == "Market" and len(compare_results) >= 2:
                opt_results = [r for r in compare_results if r["Trạng thái"] == "Optimal"]
                if len(opt_results) > 0:
                    st.markdown("#### 📊 So Sánh Hiệu Suất Điểm Chất Lượng & Chi Phí")
                    fig_market = go.Figure()
                    scenarios_names = [r["Kịch bản"] for r in opt_results]
                    quality_vals = [r["Điểm chất lượng trung bình"] for r in opt_results]
                    cost_vals = [r["Tổng chi phí tuyển dụng"] / 1000 for r in opt_results]
                    
                    fig_market.add_trace(go.Bar(
                        x=scenarios_names,
                        y=quality_vals,
                        name="Chất lượng trung bình (0-10)",
                        marker_color='#6366f1'
                    ))
                    fig_market.add_trace(go.Bar(
                        x=scenarios_names,
                        y=cost_vals,
                        name="Chi phí tuyển dụng (x1,000 USD)",
                        marker_color='#ec4899',
                        visible="legendonly"
                    ))
                    
                    fig_market.update_layout(
                        title="Chất lượng trung bình và Tổng chi phí theo biến động thị trường",
                        barmode="group",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color="#94a3b8"
                    )
                    st.plotly_chart(fig_market, use_container_width=True, key="whatif_market")
        else:
            st.info("💡 Chọn kịch bản phân tích phía trên và nhấn nút chạy tương ứng để hiển thị kết quả phân tích What-if.")
            
    with tabs[3]:
        st.markdown("### 🎲 Mô Phỏng Monte Carlo & Phân Tích Rủi Ro")
        st.write(
            "Đánh giá rủi ro, độ ổn định của phương án tuyển dụng và xác định xác suất vượt ngân sách "
            "khi chi phí nhân sự trên thị trường biến động không chắc chắn."
        )
        
        # Check if base optimal team is available
        if "base_optimal_team" not in st.session_state or "base_manual_params" not in st.session_state:
            st.warning("⚠️ Bạn chưa chạy tối ưu hóa tuyển dụng ở Tab '🎯 Tối Ưu Hóa Nguồn Lực' hoặc 'Hội thoại Chatbot'. Vui lòng chạy tối ưu hóa trước để có phương án cơ sở để chạy mô phỏng.")
        else:
            team_df = st.session_state.base_optimal_team
            base_constraints = st.session_state.base_manual_params
            
            st.info(f"💡 Mô phỏng sẽ được chạy dựa trên phương án tối ưu cơ sở hiện tại (Đã tuyển **{len(team_df)} người** với ngân sách **${base_constraints.budget:,.2f}**).")
            
            # Simulation inputs
            col_sim_cfg, col_sim_run = st.columns([1.1, 1.8])
            
            with col_sim_cfg:
                st.markdown("#### ⚙️ Cấu Hình Mô Phỏng")
                num_simulations = st.slider("Số lần chạy mô phỏng tối ưu (Runs)", min_value=10, max_value=200, value=100, step=10, help="Số lần chạy re-solve ILP. Số lần càng lớn kết quả càng chính xác nhưng thời gian chạy lâu hơn.")
                salary_volatility_pct = st.slider("Độ biến động lương thị trường (%)", min_value=5, max_value=30, value=10, step=5, help="Độ lệch chuẩn lương là X% của mức lương dự báo cơ sở (mô phỏng lương thay đổi theo phân phối Normal).")
                quality_volatility_val = st.slider("Độ lệch chất lượng (điểm)", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Độ lệch chuẩn của điểm chất lượng ứng viên (0.0 - 10.0).")
                budget_volatility_pct = st.slider("Độ biến động ngân sách (%)", min_value=0, max_value=20, value=5, step=1, help="Độ biến động ngân sách năm sau (độ lệch chuẩn).")
                
                run_sim = st.button("Bắt đầu mô phỏng 🎲", use_container_width=True, key="run_sim_btn")
                
            with col_sim_run:
                # Store simulation results in session state to persist on rerun
                if run_sim:
                    with st.spinner("Đang chạy mô phỏng Monte Carlo..."):
                        # Convert volatility percentages
                        sal_vol = salary_volatility_pct / 100.0
                        q_vol = quality_volatility_val
                        b_vol = budget_volatility_pct / 100.0
                        
                        # --- 1. Fixed Team Simulation (1000 runs, instantaneous) ---
                        fixed_costs = team_df['recruitment_cost'].values
                        sim_team_costs = []
                        over_budget_count = 0
                        
                        for _ in range(1000):
                            sim_member_costs = np.random.normal(fixed_costs, sal_vol * fixed_costs)
                            total_team_cost = np.sum(sim_member_costs)
                            sim_b = np.random.normal(base_constraints.budget, b_vol * base_constraints.budget) if b_vol > 0 else base_constraints.budget
                            
                            sim_team_costs.append(total_team_cost)
                            if total_team_cost > sim_b:
                                over_budget_count += 1
                                
                        over_budget_prob = (over_budget_count / 1000) * 100
                        mean_team_cost = np.mean(sim_team_costs)
                        sd_team_cost = np.std(sim_team_costs)
                        ci_team_cost = [mean_team_cost - 1.96 * sd_team_cost, mean_team_cost + 1.96 * sd_team_cost]
                        
                        # --- 2. Re-optimization Simulation (resolving ILP) ---
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()
                        
                        # Pre-filter candidate pool to reduce solving pool size
                        filtered_pool = df.copy()
                        if base_constraints.job_titles:
                            filtered_pool = filtered_pool[filtered_pool['job_title'].isin(base_constraints.job_titles)]
                        if getattr(base_constraints, 'allowed_experience_years', None) is not None:
                            filtered_pool = filtered_pool[filtered_pool['experience_years'].isin(base_constraints.allowed_experience_years)]
                        else:
                            if base_constraints.experience_min is not None:
                                filtered_pool = filtered_pool[filtered_pool['experience_years'] >= base_constraints.experience_min]
                            if base_constraints.experience_max is not None:
                                filtered_pool = filtered_pool[filtered_pool['experience_years'] <= base_constraints.experience_max]
                        if base_constraints.min_skill_count is not None:
                            filtered_pool = filtered_pool[filtered_pool['skills_count'] >= base_constraints.min_skill_count]
                        if base_constraints.min_certifications is not None:
                            filtered_pool = filtered_pool[filtered_pool['certifications'] >= base_constraints.min_certifications]
                            
                        if getattr(base_constraints, 'allowed_education_levels', None):
                            filtered_pool = filtered_pool[filtered_pool['education_level'].isin(base_constraints.allowed_education_levels)]
                        elif base_constraints.education_levels and base_constraints.education_ratio_min is None and base_constraints.education_ratio_max is None:
                            filtered_pool = filtered_pool[filtered_pool['education_level'].isin(base_constraints.education_levels)]
                            
                        remote_map = {'Remote': 'Yes', 'On-site': 'No', 'Hybrid': 'Hybrid'}
                        if getattr(base_constraints, 'allowed_remote_types', None):
                            db_allowed_remote = [remote_map[r] for r in base_constraints.allowed_remote_types if r in remote_map]
                            if db_allowed_remote:
                                filtered_pool = filtered_pool[filtered_pool['remote_work'].isin(db_allowed_remote)]
                        elif base_constraints.remote_types and base_constraints.remote_ratio_min is None and base_constraints.remote_ratio_max is None:
                            db_remote_types = [remote_map[r] for r in base_constraints.remote_types if r in remote_map]
                            if db_remote_types:
                                filtered_pool = filtered_pool[filtered_pool['remote_work'].isin(db_remote_types)]
                                
                        filtered_pool = filtered_pool.sort_values(by="quality_score", ascending=False).head(1000)
                        filtered_pool = filtered_pool.reset_index(drop=True)
                        
                        candidate_selection_counts = {row['id']: 0 for _, row in filtered_pool.iterrows()}
                        candidate_info = {row['id']: row for _, row in filtered_pool.iterrows()}
                        
                        active_job_titles = base_constraints.job_titles if base_constraints.job_titles else ["AI Engineer", "Data Scientist", "Data Analyst", "Backend Developer", "Frontend Developer", "Machine Learning Engineer"]
                        job_title_counts = {title: [] for title in active_job_titles}
                        
                        reopt_results = []
                        success_count = 0
                        
                        for r in range(num_simulations):
                            status_text.text(f"Đang chạy mô phỏng tối ưu hóa {r+1}/{num_simulations}...")
                            df_run = filtered_pool.copy()
                            
                            if sal_vol > 0:
                                df_run['recruitment_cost'] = np.random.normal(df_run['recruitment_cost'], sal_vol * df_run['recruitment_cost'])
                            if q_vol > 0:
                                df_run['quality_score'] = np.random.normal(df_run['quality_score'], q_vol)
                                df_run['quality_score'] = np.clip(df_run['quality_score'], 0.0, 10.0)
                                
                            sim_b = np.random.normal(base_constraints.budget, b_vol * base_constraints.budget) if b_vol > 0 else base_constraints.budget
                            run_params = base_constraints.copy(update={"budget": sim_b})
                            
                            res_df = solve_recruitment_ilp_engine(df_run, run_params)
                            
                            if res_df is not None:
                                success_count += 1
                                total_cost = res_df['recruitment_cost'].sum()
                                avg_q = res_df['quality_score'].mean()
                                total_q = res_df['quality_score'].sum()
                                hc = len(res_df)
                                
                                reopt_results.append({
                                    "Run": r + 1,
                                    "Status": "Optimal",
                                    "Total Cost": total_cost,
                                    "Avg Quality": avg_q,
                                    "Total Quality": total_q,
                                    "Headcount": hc
                                })
                                
                                # Track candidate selection
                                for cid in res_df['id']:
                                    if cid in candidate_selection_counts:
                                        candidate_selection_counts[cid] += 1
                                        
                                # Track job title counts
                                counts = res_df['job_title'].value_counts().to_dict()
                                for title in job_title_counts:
                                    job_title_counts[title].append(counts.get(title, 0))
                            else:
                                reopt_results.append({
                                    "Run": r + 1,
                                    "Status": "Infeasible",
                                    "Total Cost": 0.0,
                                    "Avg Quality": 0.0,
                                    "Total Quality": 0.0,
                                    "Headcount": 0
                                })
                            progress_bar.progress((r + 1) / num_simulations)
                            
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Store everything in session state
                        st.session_state.mc_run = True
                        st.session_state.mc_success_rate = (success_count / num_simulations) * 100
                        st.session_state.mc_over_budget_prob = over_budget_prob
                        st.session_state.mc_mean_team_cost = mean_team_cost
                        st.session_state.mc_sd_team_cost = sd_team_cost
                        st.session_state.mc_ci_team_cost = ci_team_cost
                        st.session_state.mc_reopt_df = pd.DataFrame(reopt_results)
                        st.session_state.mc_salary_vol = salary_volatility_pct
                        st.session_state.mc_quality_vol = quality_volatility_val
                        st.session_state.mc_budget_vol = budget_volatility_pct
                        
                        # Selection frequency
                        sel_freq = []
                        for cid, count in candidate_selection_counts.items():
                            if count > 0:
                                info = candidate_info[cid]
                                sel_freq.append({
                                    "Chức danh": info['job_title'],
                                    "Số năm KN": info['experience_years'],
                                    "Học vấn": info['education_level'],
                                    "Remote": info['remote_work'],
                                    "Chất lượng gốc": f"{info['quality_score']:.2f}",
                                    "Chi phí gốc": f"${info['recruitment_cost']:,.2f}",
                                    "Tần suất xuất hiện": count / success_count if success_count > 0 else 0
                                })
                        st.session_state.mc_candidate_frequency = sorted(sel_freq, key=lambda x: x["Tần suất xuất hiện"], reverse=True)
                        
                        # Job title averages
                        title_avg = {}
                        for title, arr in job_title_counts.items():
                            title_avg[title] = np.mean(arr) if len(arr) > 0 else 0
                        st.session_state.mc_job_title_frequency = title_avg

                # Display Results
                if st.session_state.get("mc_run", False):
                    st.markdown("### 📊 Kết Quả Mô Phỏng Monte Carlo")
                    
                    # Columns for Key Risk Metrics
                    kcol1, kcol2, kcol3 = st.columns(3)
                    with kcol1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Xác suất vượt ngân sách (Đội hình gốc)</div>
                            <div class="metric-value" style="color: {'#ef4444' if st.session_state.mc_over_budget_prob > 15.0 else '#f59e0b' if st.session_state.mc_over_budget_prob > 5.0 else '#10b981'}">{st.session_state.mc_over_budget_prob:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with kcol2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Tỷ lệ khả thi tuyển dụng (Re-optimization)</div>
                            <div class="metric-value">{st.session_state.mc_success_rate:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with kcol3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-title">Chi phí TB đội hình gốc</div>
                            <div class="metric-value">${st.session_state.mc_mean_team_cost:,.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Detailed Stats
                    df_mc = st.session_state.mc_reopt_df
                    df_mc_opt = df_mc[df_mc["Status"] == "Optimal"]
                    
                    st.markdown(f"**Khoảng tin cậy 95% của Chi phí Đội hình gốc**: **[${st.session_state.mc_ci_team_cost[0]:,.2f} - ${st.session_state.mc_ci_team_cost[1]:,.2f}]** (Độ lệch chuẩn: **${st.session_state.mc_sd_team_cost:,.2f}**)")
                    
                    if len(df_mc_opt) > 0:
                        # Display Charts
                        rcol1, rcol2 = st.columns(2)
                        with rcol1:
                            fig_mc_cost = px.histogram(df_mc_opt, x="Total Cost", title="Phân phối Tổng chi phí khi Tối ưu lại", color_discrete_sequence=['#6366f1'])
                            fig_mc_cost.add_vline(x=base_constraints.budget, line_dash="dash", line_color="red", annotation_text="Ngân sách cơ sở")
                            fig_mc_cost.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                            st.plotly_chart(fig_mc_cost, use_container_width=True, key="mc_cost")
                            
                            fig_mc_q = px.histogram(df_mc_opt, x="Avg Quality", title="Phân phối Chất lượng TB khi Tối ưu lại", color_discrete_sequence=['#ec4899'])
                            fig_mc_q.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                            st.plotly_chart(fig_mc_q, use_container_width=True, key="mc_quality")
                            
                        with rcol2:
                            # Job title frequencies
                            title_freqs = st.session_state.mc_job_title_frequency
                            df_tf = pd.DataFrame(list(title_freqs.items()), columns=["Chức danh", "SL tuyển trung bình"])
                            fig_mc_titles = px.bar(df_tf, x="Chức danh", y="SL tuyển trung bình", title="Số lượng tuyển trung bình theo Chức danh", color_discrete_sequence=['#10b981'])
                            fig_mc_titles.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#94a3b8")
                            st.plotly_chart(fig_mc_titles, use_container_width=True, key="mc_titles")
                            
                        # Candidate selection stability table
                        st.markdown("#### ⚓ Danh sách Ứng viên Mỏ neo (Anchor Candidates)")
                        st.write("Ứng viên được chọn với tần suất rất cao (>= 75%) qua các lần mô phỏng, thể hiện độ tin cậy vượt trội bất kể biến động thị trường.")
                        
                        cand_freq = st.session_state.mc_candidate_frequency
                        df_cand_freq = pd.DataFrame(cand_freq)
                        if not df_cand_freq.empty:
                            df_cand_freq["Tần suất xuất hiện"] = df_cand_freq["Tần suất xuất hiện"].apply(lambda x: f"{x*100:.1f}%")
                            df_cand_freq.index = range(1, len(df_cand_freq) + 1)
                            df_cand_freq.index.name = "STT"
                            st.dataframe(df_cand_freq.head(10), use_container_width=True)
                        else:
                            st.info("Không có ứng viên nào được chọn.")
                            
                        # Risk Report Text
                        st.markdown("---")
                        st.markdown("#### 📄 Báo Cáo Đánh Giá Rủi Ro Monte Carlo")
                        
                        def generate_risk_report():
                            ob = st.session_state.mc_over_budget_prob
                            sr = st.session_state.mc_success_rate
                            sal_v = st.session_state.mc_salary_vol
                            q_v = st.session_state.mc_quality_vol
                            b_v = st.session_state.mc_budget_vol
                            
                            report = (
                                f"=== BÁO CÁO PHÂN TÍCH RỦI RO & ĐỘ ỔN ĐỊNH DSS ===\n"
                                f"Phương pháp: Mô phỏng Monte Carlo (salary_volatility={sal_v}%, quality_volatility={q_v} điểm, budget_volatility={b_v}%)\n\n"
                                f"--- 1. RỦI RO NGÂN SÁCH (Fixed Team Budget Risk) ---\n"
                                f"- Xác suất vượt ngân sách của đội hình tối ưu cơ sở là: {ob:.1f}%\n"
                                f"  * Với chi phí lương thị trường biến động {sal_v}%, chi phí thực tế của đội hình đã chọn sẽ dao động có độ lệch chuẩn là ${st.session_state.mc_sd_team_cost:,.2f}.\n"
                                f"  * Khoảng tin cậy 95% của tổng chi phí: [${st.session_state.mc_ci_team_cost[0]:,.2f} - ${st.session_state.mc_ci_team_cost[1]:,.2f}].\n"
                            )
                            if ob > 20.0:
                                report += f"  * ĐÁNH GIÁ: RỦI RO CAO! Kế hoạch tuyển dụng này có khả năng vượt ngân sách lớn. Khuyến nghị CHRO nên chuẩn bị thêm khoản dự phòng tài chính khoảng {(ob/10):.1f}% của ngân sách gốc để bảo đảm tuyển đủ đội hình.\n"
                            elif ob > 5.0:
                                report += f"  * ĐÁNH GIÁ: RỦI RO TRUNG BÌNH. Kế hoạch tương đối an toàn, tuy nhiên vẫn cần theo dõi biến động thị trường lương.\n"
                            else:
                                report += f"  * ĐÁNH GIÁ: RỦI RO THẤP. Phương án tuyển dụng vô cùng an toàn về mặt chi phí.\n"
                                
                            report += (
                                f"\n--- 2. ĐỘ ỔN ĐỊNH CỦA BÀI TOÁN TỐI ƯU (Re-optimization Feasibility) ---\n"
                                f"- Tỷ lệ giải bài toán thành công khi biến động: {sr:.1f}%\n"
                            )
                            if sr < 80.0:
                                report += (
                                    f"  * ĐÁNH GIÁ: ĐỘ ỔN ĐỊNH THẤP ({sr:.1f}% khả thi). Các ràng buộc nâng cao hiện tại quá chặt chẽ (như Senior/Remote/Học vấn) khiến bài toán rất dễ rơi vào trạng thái không khả thi (Infeasible) khi lương ứng viên biến động tăng.\n"
                                    f"  * Khuyến nghị: CHRO nên xem xét nới lỏng bớt các tiêu chí không bắt buộc ở Tab 1 để tăng tính linh hoạt cho quy trình tuyển dụng.\n"
                                )
                            else:
                                report += "  * ĐÁNH GIÁ: ĐỘ ỔN ĐỊNH CAO. Mô hình tối ưu ổn định vượt trội trước các biến động ngẫu nhiên của thị trường.\n"
                                
                            report += "\n--- 3. DANH SÁCH ỨNG VIÊN MỎ NEO (Anchor Candidates) ---\n"
                            anchors = [c for c in cand_freq if c["Tần suất xuất hiện"] >= 0.8]
                            if anchors:
                                report += f"Phát hiện có {len(anchors)} ứng viên 'Mỏ neo' được chọn với tần suất >= 80%:\n"
                                for a in anchors[:5]:
                                    report += f"  * Ứng viên {a['Chức danh']} ({a['Số năm KN']} năm KN, {a['Học vấn']}, Remote: {a['Remote']}, Chi phí: {a['Chi phí gốc']}) - Tần suất chọn: {a['Tần suất xuất hiện']*100:.0f}%\n"
                                report += "Khuyến nghị: Những ứng viên này là những nhân sự 'bắt buộc phải tuyển' (Must-hire) vì họ luôn đem lại hiệu quả chi phí/chất lượng tối ưu nhất bất kể lương thị trường biến đổi thế nào.\n"
                            else:
                                report += "Không phát hiện ứng viên mỏ neo nổi bật nào có tần suất chọn trên 80%. Đội hình tuyển dụng có độ tùy biến thay đổi linh hoạt cao.\n"
                                
                            report += "\n=== HẾT BÁO CÁO ==="
                            return report
                            
                        report_text = generate_risk_report()
                        st.text_area("Nội dung báo cáo rủi ro", report_text, height=220, key="risk_report_textarea")
                        
                        # Download button
                        st.download_button("Tải Báo Cáo Đánh Giá Rủi Ro (.txt) 📥", report_text, file_name="Risk_Analysis_Report.txt", key="download_risk_report")
                    else:
                        st.warning("⚠️ Không có kịch bản tối ưu khả thi nào được tìm thấy trong số các lần mô phỏng tái tối ưu hóa. Vui lòng nới lỏng các ràng buộc cố định ở Tab 1.")
                else:
                    st.info("💡 Thiết lập các thông số bên trái và click **Bắt đầu mô phỏng** để phân tích rủi ro.")
