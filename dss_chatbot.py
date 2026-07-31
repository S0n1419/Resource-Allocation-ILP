import re
import json
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

class ExperienceGroupConstraints(BaseModel):
    min_count: Optional[int] = Field(default=None, description="Absolute minimum count.")
    max_count: Optional[int] = Field(default=None, description="Absolute maximum count.")
    min_ratio: Optional[float] = Field(default=None, description="Minimum ratio (0.0-1.0).")
    max_ratio: Optional[float] = Field(default=None, description="Maximum ratio (0.0-1.0).")

class ExperienceGroups(BaseModel):
    Fresher: Optional[ExperienceGroupConstraints] = Field(default=None)
    Junior: Optional[ExperienceGroupConstraints] = Field(default=None)
    Mid: Optional[ExperienceGroupConstraints] = Field(default=None)
    Senior: Optional[ExperienceGroupConstraints] = Field(default=None)
    Expert: Optional[ExperienceGroupConstraints] = Field(default=None)

class HighSkillsConstraints(BaseModel):
    threshold: int = Field(description="Skill count threshold.")
    min_ratio: Optional[float] = Field(default=None, description="Minimum ratio.")
    max_ratio: Optional[float] = Field(default=None, description="Maximum ratio.")

class HighCertsConstraints(BaseModel):
    threshold: int = Field(description="Certification count threshold.")
    min_ratio: Optional[float] = Field(default=None, description="Minimum ratio.")
    max_ratio: Optional[float] = Field(default=None, description="Maximum ratio.")

class JobTitleLimitsConstraints(BaseModel):
    min: Optional[int] = Field(default=None, description="Minimum headcount limit.")
    max: Optional[int] = Field(default=None, description="Maximum headcount limit.")

# 1. Pydantic schema for structured recruitment constraints
class RecruitmentConstraints(BaseModel):
    budget: float = Field(
        description="Total budget for recruitment in USD (e.g. 1500000)"
    )
    num_employees: int = Field(
        description="Total number of people to hire (e.g. 10)"
    )
    job_titles: Optional[List[str]] = Field(
        default=None,
        description="List of job titles allowed to recruit"
    )
    job_title_limits: Optional[Dict[str, JobTitleLimitsConstraints]] = Field(
        default=None,
        description="Absolute headcount limits per job title"
    )
    job_title_max_ratio: Optional[Dict[str, float]] = Field(
        default=None,
        description="Maximum ratio limit per job title"
    )
    experience_min: Optional[int] = Field(
        default=None,
        description="Minimum experience years required for any candidate"
    )
    experience_max: Optional[int] = Field(
        default=None,
        description="Maximum experience years allowed for any candidate"
    )
    allowed_experience_years: Optional[List[int]] = Field(
        default=None,
        description="Set of accepted experience years (J_allow)"
    )
    experience_groups: Optional[ExperienceGroups] = Field(
        default=None,
        description="Experience structure constraints"
    )
    allowed_education_levels: Optional[List[str]] = Field(
        default=None,
        description="List of education levels allowed to recruit (E_allow - Mục 5.5.1)"
    )
    education_levels: Optional[List[str]] = Field(
        default=None,
        description="Target education levels for the ratio constraints (E_h - Mục 5.5.2)"
    )
    education_ratio_min: Optional[float] = Field(
        default=None,
        description="Minimum ratio of candidates having specified education levels"
    )
    education_ratio_max: Optional[float] = Field(
        default=None,
        description="Maximum ratio of candidates having specified education levels"
    )
    min_skill_count: Optional[int] = Field(
        default=None,
        description="Minimum skills count required for any candidate"
    )
    skills_high: Optional[HighSkillsConstraints] = Field(
        default=None,
        description="Ratio constraint for candidates with high skill count"
    )
    min_certifications: Optional[int] = Field(
        default=None,
        description="Minimum certifications count required for any candidate"
    )
    certifications_high: Optional[HighCertsConstraints] = Field(
        default=None,
        description="Ratio constraint for candidates with high certification count"
    )
    allowed_remote_types: Optional[List[str]] = Field(
        default=None,
        description="List of remote work types allowed (R_allow - Mục 5.8.1)"
    )
    remote_types: Optional[List[str]] = Field(
        default=None,
        description="Target remote work types for the ratio constraints"
    )
    remote_ratio_min: Optional[float] = Field(
        default=None,
        description="Minimum ratio of remote work types"
    )
    remote_ratio_max: Optional[float] = Field(
        default=None,
        description="Maximum ratio of remote work types"
    )
    min_avg_quality_score: Optional[float] = Field(
        default=None,
        description="Minimum average quality score of hired team"
    )

# 2. Mock / Rule-based parser using Regular Expressions
def mock_parse_query(query: str) -> RecruitmentConstraints:
    """
    Parses a Vietnamese natural language query using regex rules to extract ILP parameters.
    """
    query_lower = query.lower()
    params = {}
    
    # Extract budget
    budget_match = re.search(r'ngân\s+sách\s*[:\-]?\s*([\d\.,]+)\s*(triệu|m|tr)?', query_lower)
    if budget_match:
        val_str = budget_match.group(1).replace(',', '.')
        try:
            val = float(val_str)
            unit = budget_match.group(2)
            if unit in ['triệu', 'm', 'tr']:
                val *= 1000000
            elif val < 10000:
                val *= 1000000
            params['budget'] = val
        except ValueError:
            pass

    # Extract headcount
    headcount_match = re.search(r'(tuyển|cần tuyển|headcount|nhu cầu)\s*([\d]+)\s*(người|nhân sự|ứng viên|vị trí)?', query_lower)
    if headcount_match:
        try:
            params['num_employees'] = int(headcount_match.group(2))
        except ValueError:
            pass

    # Extract allowed titles
    all_titles = ["AI Engineer", "Data Scientist", "Data Analyst", "Backend Developer", "Frontend Developer", "Machine Learning Engineer"]
    found_titles = []
    for title in all_titles:
        if title.lower() in query_lower:
            found_titles.append(title)
    if found_titles:
        params['job_titles'] = found_titles

    # Extract min/max experience
    min_exp_match = re.search(r'(kinh\s+nghiệm\s+tối\s+thiểu|kinh\s+nghiệm\s+ít\s+nhất|tối\s+thiểu|ít\s+nhất)\s+(\d+)\s+năm', query_lower)
    if min_exp_match:
        params['experience_min'] = int(min_exp_match.group(2))

    max_exp_match = re.search(r'(kinh\s+nghiệm\s+tối\s+đa|tối\s+đa|không\s+quá)\s+(\d+)\s+năm', query_lower)
    if max_exp_match:
        params['experience_max'] = int(max_exp_match.group(2))

    # Extract min skills
    min_skills_match = re.search(r'kỹ\s+năng\s+tối\s+thiểu\s+(\d+)', query_lower)
    if min_skills_match:
        params['min_skill_count'] = int(min_skills_match.group(1))

    # Extract min certs
    min_certs_match = re.search(r'chứng\s+chỉ\s+tối\s+thiểu\s+(\d+)', query_lower)
    if min_certs_match:
        params['min_certifications'] = int(min_certs_match.group(1))

    # Extract experience groups
    exp_groups = {}
    
    # Senior ratio
    senior_ratio_match = re.search(r'(ít\s+nhất|tối\s+thiểu)\s+(\d+)%\s+(là\s+)?senior', query_lower)
    if senior_ratio_match:
        exp_groups['Senior'] = {'min_ratio': float(senior_ratio_match.group(2)) / 100.0}
    
    # Fresher ratio
    fresher_ratio_match = re.search(r'(không\s+quá|tối\s+đa)\s+(\d+)%\s+(là\s+)?fresher', query_lower)
    if fresher_ratio_match:
        if 'Fresher' not in exp_groups:
            exp_groups['Fresher'] = {}
        exp_groups['Fresher']['max_ratio'] = float(fresher_ratio_match.group(2)) / 100.0
        
    # Expert count
    expert_count_match = re.search(r'đúng\s+(\d+)\s+(người\s+)?(là\s+)?expert', query_lower)
    if expert_count_match:
        count_val = int(expert_count_match.group(1))
        exp_groups['Expert'] = {'min_count': count_val, 'max_count': count_val}
        
    if exp_groups:
        params['experience_groups'] = exp_groups

    # Extract allowed education levels (hard filter)
    if "chỉ tuyển" in query_lower or "chỉ nhận" in query_lower or "chỉ lấy" in query_lower or "yêu cầu bằng" in query_lower or "yêu cầu trình độ" in query_lower:
        matched_levels = []
        if "master" in query_lower or "thạc sĩ" in query_lower:
            matched_levels.append("Master")
        if "phd" in query_lower or "tiến sĩ" in query_lower:
            matched_levels.append("PhD")
        if "bachelor" in query_lower or "đại học" in query_lower or "cử nhân" in query_lower:
            matched_levels.append("Bachelor")
        if "diploma" in query_lower or "cao đẳng" in query_lower:
            matched_levels.append("Diploma")
        if "high school" in query_lower or "trung học" in query_lower:
            matched_levels.append("High School")
        if matched_levels:
            params['allowed_education_levels'] = matched_levels

    if 'allowed_education_levels' not in params:
        if "đại học trở lên" in query_lower or "cử nhân trở lên" in query_lower:
            params['allowed_education_levels'] = ["Bachelor", "Master", "PhD"]
        elif "thạc sĩ trở lên" in query_lower:
            params['allowed_education_levels'] = ["Master", "PhD"]

    # Extract education ratio
    edu_ratio_match = re.search(r'(ít\s+nhất|tối\s+thiểu)\s+(\d+)%\s+(là\s+)?(master|phd|thạc\s+sĩ|tiến\s+sĩ|học\s+vấn)', query_lower)
    if edu_ratio_match:
        try:
            params['education_levels'] = ["Master", "PhD"]
            params['education_ratio_min'] = float(edu_ratio_match.group(2)) / 100.0
        except ValueError:
            pass

    # Extract allowed remote types (hard filter)
    if "chỉ làm" in query_lower or "chỉ tuyển hình thức" in query_lower or "chỉ nhận hình thức" in query_lower:
        matched_remote = []
        if "remote" in query_lower or "từ xa" in query_lower:
            matched_remote.append("Remote")
        if "hybrid" in query_lower:
            matched_remote.append("Hybrid")
        if "on-site" in query_lower or "lên văn phòng" in query_lower:
            matched_remote.append("On-site")
        if matched_remote:
            params['allowed_remote_types'] = matched_remote

    # Extract remote/hybrid ratios
    hybrid_match = re.search(r'(tối\s+thiểu|ít\s+nhất)\s+(\d+)%\s+(làm\s+)?hybrid', query_lower)
    if hybrid_match:
        try:
            params['remote_types'] = ["Hybrid"]
            params['remote_ratio_min'] = float(hybrid_match.group(2)) / 100.0
        except ValueError:
            pass

    remote_match = re.search(r'(tối\s+đa|không\s+quá)\s+(\d+)%\s+(làm\s+)?remote', query_lower)
    if remote_match:
        try:
            if 'remote_types' in params:
                if "Remote" not in params['remote_types']:
                    params['remote_types'].append("Remote")
                params['remote_ratio_max'] = float(remote_match.group(2)) / 100.0
            else:
                params['remote_types'] = ["Remote"]
                params['remote_ratio_max'] = float(remote_match.group(2)) / 100.0
        except ValueError:
            pass

    # Safe defaults for required schema properties if extraction fails
    if 'budget' not in params:
        params['budget'] = 1500000.0
    if 'num_employees' not in params:
        params['num_employees'] = 10

    return RecruitmentConstraints(**params)

# 3. Helpers to extract and clean schema for Gemini API (removes unsupported "default" fields)
def get_gemini_schema(model_class):
    schema = model_class.model_json_schema()
    
    def clean_schema(d):
        if not isinstance(d, dict):
            return d
        
        unsupported = ["default", "title", "examples", "definitions", "$defs"]
        for key in unsupported:
            d.pop(key, None)
            
        if "anyOf" in d:
            subschemas = d.pop("anyOf")
            types = [s.get("type") for s in subschemas if "type" in s]
            non_null_types = [t for t in types if t != "null"]
            if non_null_types:
                d["type"] = non_null_types[0]
            if "null" in types:
                d["nullable"] = True
                
        for k, v in list(d.items()):
            if isinstance(v, dict):
                d[k] = clean_schema(v)
            elif isinstance(v, list):
                d[k] = [clean_schema(item) if isinstance(item, dict) else item for item in v]
                
        return d

    return clean_schema(schema)

def dict_to_proto_schema(d):
    import google.generativeai as genai
    if not isinstance(d, dict):
        return d
        
    kwargs = {}
    
    if "type" in d:
        kwargs["type"] = d["type"].upper()
        
    if "description" in d:
        kwargs["description"] = d["description"]
        
    if "nullable" in d:
        kwargs["nullable"] = d["nullable"]
        
    if "items" in d:
        kwargs["items"] = dict_to_proto_schema(d["items"])
        
    if "properties" in d:
        kwargs["properties"] = {k: dict_to_proto_schema(v) for k, v in d["properties"].items()}
        
    if "required" in d:
        kwargs["required"] = d["required"]
        
    return genai.protos.Schema(**kwargs)

# 4. Gemini API parser for extracting structured parameters
def parse_query_with_gemini(query: str, api_key: str) -> RecruitmentConstraints:
    """
    Calls Gemini API to parse natural language queries into a structured RecruitmentConstraints object.
    """
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    system_prompt = (
        "Bạn là chuyên gia trích xuất thông tin tuyển dụng.\n"
        "Nhiệm vụ: Đọc câu hỏi của người dùng và trích xuất các ràng buộc (constraints) thành tham số JSON cho function/schema tuyển dụng.\n\n"
        "QUY TẮC XỬ LÝ (RULES - Quan trọng):\n"
        "1. Nếu người dùng nói \"đúng X người\" cho một nhóm (ví dụ: \"đúng 2 Expert\") -> hãy đặt cả \"min_count\": X và \"max_count\": X cho nhóm đó.\n"
        "2. Nếu người dùng nói \"ít nhất X%\" -> tính ra tỷ lệ thập phân: \"X\" / 100 (vd: 30% -> 0.3).\n"
        "3. Nếu người dùng nói \"không quá X%\" -> tính ra tỷ lệ thập phân và đặt vào \"max_ratio\".\n"
        "4. Phân biệt bộ lọc cứng và nhóm tính tỷ lệ:\n"
        "   - Nếu người dùng yêu cầu giới hạn cứng trình độ (ví dụ: \"chỉ tuyển cử nhân trở lên\", \"chỉ nhận Master, PhD\") -> điền vào \"allowed_education_levels\".\n"
        "   - Nếu người dùng yêu cầu tỷ lệ cho một nhóm trình độ học vấn (ví dụ: \"ít nhất 30% là master/phd\") -> điền nhóm đó vào \"education_levels\" và đặt tỷ lệ vào \"education_ratio_min\"/\"max\". KHÔNG ĐIỀN vào \"allowed_education_levels\" trừ khi họ ghi rõ chỉ tuyển nhóm đó.\n"
        "   - Tương tự cho hình thức làm việc: \"allowed_remote_types\" là bộ lọc cứng, \"remote_types\" là nhóm tính tỷ lệ.\n"
        "5. Nếu người dùng chỉ đề cập đến một danh sách chức danh được tuyển -> điền vào \"job_titles\".\n"
        "6. Nếu người dùng không đề cập đến một trường nào đó, KHÔNG ĐƯỢC TỰ Ý ĐIỀN giá trị mặc định. Hãy để trống (null).\n"
        "7. Giữ nguyên đơn vị tiền tệ (USD). Nếu người dùng nói \"1.5 triệu USD\" hoặc \"1.5 triệu đô\" -> hãy tính 1500000.\n"
        "8. Hãy chú ý đến ngữ cảnh phân loại kinh nghiệm:\n"
        "   - \"Fresher\" là dưới 1 năm kinh nghiệm (< 1 năm).\n"
        "   - \"Junior\" là 1-3 năm kinh nghiệm.\n"
        "   - \"Mid\" là 3-5 năm kinh nghiệm.\n"
        "   - \"Senior\" là 5-8 năm kinh nghiệm.\n"
        "   - \"Expert\" là từ 8 năm kinh nghiệm trở lên (>= 8 năm)."
    )
    
    prompt = f"{system_prompt}\n\nUser recruitment request: {query}"
    
    # Get clean schema representation as a protobuf Schema message
    clean_dict = get_gemini_schema(RecruitmentConstraints)
    proto_schema = dict_to_proto_schema(clean_dict)
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            response_schema=proto_schema,
        ),
    )
    
    data = json.loads(response.text)
    return RecruitmentConstraints(**data)

# 5. Bot response generator
def format_bot_response(query: str, extracted_params: RecruitmentConstraints, results_df, budget: float, headcount: int) -> str:
    """
    Formats the chatbot's response in markdown based on the result of the optimization.
    """
    # Format constraint strings for display
    senior_constraints = (
        extracted_params.experience_groups.Senior 
        if extracted_params.experience_groups and hasattr(extracted_params.experience_groups, 'Senior') and extracted_params.experience_groups.Senior 
        else None
    )
    senior_req = "Không yêu cầu"
    if senior_constraints:
        parts = []
        if hasattr(senior_constraints, 'min_ratio') and senior_constraints.min_ratio is not None:
            parts.append(f"tối thiểu {senior_constraints.min_ratio*100:.0f}%")
        if hasattr(senior_constraints, 'max_ratio') and senior_constraints.max_ratio is not None:
            parts.append(f"tối đa {senior_constraints.max_ratio*100:.0f}%")
        if hasattr(senior_constraints, 'min_count') and senior_constraints.min_count is not None:
            parts.append(f"ít nhất {senior_constraints.min_count} người")
        if hasattr(senior_constraints, 'max_count') and senior_constraints.max_count is not None:
            parts.append(f"tối đa {senior_constraints.max_count} người")
        if parts:
            senior_req = ", ".join(parts)

    edu_req = f"tối thiểu {extracted_params.education_ratio_min*100:.0f}%" if extracted_params.education_ratio_min is not None else "Không yêu cầu"
    
    hybrid_req = "Không yêu cầu"
    if extracted_params.remote_ratio_min is not None and extracted_params.remote_types and "Hybrid" in extracted_params.remote_types:
        hybrid_req = f"tối thiểu {extracted_params.remote_ratio_min*100:.0f}%"
        
    remote_req = "Không giới hạn"
    if extracted_params.remote_ratio_max is not None and extracted_params.remote_types and "Remote" in extracted_params.remote_types:
        remote_req = f"tối đa {extracted_params.remote_ratio_max*100:.0f}%"
    
    allowed_edu_str = ', '.join(extracted_params.allowed_education_levels) if extracted_params.allowed_education_levels else 'Bất kỳ'
    allowed_rem_str = ', '.join(extracted_params.allowed_remote_types) if extracted_params.allowed_remote_types else 'Bất kỳ'

    if results_df is not None:
        total_cost = results_df["recruitment_cost"].sum()
        avg_quality = results_df["quality_score"].mean()
        
        # Calculate structure percentages
        title_counts = results_df["job_title"].value_counts()
        total_hired = len(results_df)
        
        title_str = ", ".join([f"{title}: {count}" for title, count in title_counts.items()])
        
        hybrid_count = sum(results_df['remote_work'] == 'Hybrid')
        remote_count = sum(results_df['remote_work'] == 'Yes')
        hybrid_ratio = hybrid_count / total_hired if total_hired > 0 else 0
        remote_ratio = remote_count / total_hired if total_hired > 0 else 0
        
        # Calculate Senior (5-8 years)
        senior_count = sum((results_df['experience_years'] >= 5) & (results_df['experience_years'] < 8))
        senior_ratio = senior_count / total_hired if total_hired > 0 else 0
        
        edu_count = sum(results_df['education_level'].isin(extracted_params.education_levels)) if extracted_params.education_levels else 0
        edu_ratio = edu_count / total_hired if total_hired > 0 else 0
        
        response = (
            f"### 🎉 Đã tìm thấy phương án tuyển dụng tối ưu!\n\n"
            f"**Thông số đầu vào trích xuất được:**\n"
            f"- Ngân sách tối đa: **${budget:,.2f}**\n"
            f"- Số lượng cần tuyển: **{headcount} người**\n"
            f"- Các chức danh: {', '.join(extracted_params.job_titles) if extracted_params.job_titles else 'Tất cả các chức danh'}\n"
            f"- Học vấn được phép: **{allowed_edu_str}**\n"
            f"- Làm việc được phép: **{allowed_rem_str}**\n\n"
            f"**Kết quả tối ưu hóa:**\n"
            f"- **Tổng chi phí thực tế**: **${total_cost:,.2f}** (Tiết kiệm được **${budget - total_cost:,.2f}**)\n"
            f"- **Điểm chất lượng trung bình (Quality Score)**: **{avg_quality:.2f} / 10.0**\n"
            f"- **Cơ cấu chức danh**: {title_str}\n"
            f"- **Cơ cấu kinh nghiệm (5-8 năm)**: {senior_ratio*100:.1f}% Senior (yêu cầu: {senior_req})\n"
            f"- **Cơ cấu học vấn ({'/'.join(extracted_params.education_levels) if extracted_params.education_levels else 'Bất kỳ'})**: {edu_ratio*100:.1f}% (yêu cầu: {edu_req})\n"
            f"- **Hình thức làm việc**: {hybrid_ratio*100:.1f}% Hybrid (yêu cầu: {hybrid_req}), {remote_ratio*100:.1f}% Remote (yêu cầu: {remote_req})\n\n"
            f"Dưới đây là danh sách ứng viên chi tiết được lựa chọn tối ưu. Bạn có thể xem bảng kết quả và các biểu đồ cơ cấu nhân sự bên dưới."
        )
    else:
        # Infeasible explanation helper
        response = (
            f"### ❌ Không tìm được phương án khả thi!\n\n"
            f"**Thông số đầu vào trích xuất được:**\n"
            f"- Ngân sách: **${budget:,.2f}**\n"
            f"- Số lượng cần tuyển: **{headcount} người**\n"
            f"- Học vấn được phép: **{allowed_edu_str}**\n"
            f"- Làm việc được phép: **{allowed_rem_str}**\n"
            f"- Ràng buộc kinh nghiệm Senior (5-8 năm): {senior_req}\n"
            f"- Ràng buộc học vấn: {edu_req}\n"
            f"- Ràng buộc làm việc: Hybrid {hybrid_req}, Remote {remote_req}\n\n"
            f"**Lời khuyên từ DSS:**\n"
            f"1. **Tăng ngân sách**: Mức ngân sách hiện tại có thể quá thấp so với chi phí tuyển dụng ước tính của các ứng viên thỏa mãn các tiêu chuẩn khắt khe (ví dụ: Senior có học vị cao).\n"
            f"2. **Nới lỏng ràng buộc cơ cấu**: Giảm bớt tỷ lệ Senior, tỷ lệ Master/PhD hoặc mở rộng danh sách chức danh được phép tuyển dụng.\n"
            f"3. **Giảm số lượng tuyển (Headcount)**: Nếu ngân sách không đổi, giảm bớt số lượng người tuyển để có thể đáp ứng chất lượng."
        )
    return response

# 6. Post-processing & Validation
def validate_and_fix(constraints: RecruitmentConstraints) -> Tuple[RecruitmentConstraints, Optional[str]]:
    """
    Validates and fixes/normalizes extracted constraints.
    Returns a tuple of (constraints, error_message).
    """
    error_msg = None
    
    # Validate required fields
    if constraints.budget is None or constraints.budget <= 0:
        error_msg = "⚠️ Hệ thống không thể xác định ngân sách tuyển dụng. Vui lòng ghi rõ ngân sách (ví dụ: 'ngân sách 1.5 triệu USD' hoặc 'budget 1.5M')."
    elif constraints.num_employees is None or constraints.num_employees <= 0:
        error_msg = "⚠️ Hệ thống không thể xác định số lượng nhân sự cần tuyển. Vui lòng ghi rõ số lượng cần tuyển (ví dụ: 'tuyển 10 người' hoặc 'nhu cầu 10 nhân sự')."
        
    # Normalize allowed_experience_years if not explicitly set but min/max is available
    if getattr(constraints, 'allowed_experience_years', None) is None:
        if constraints.experience_min is not None or constraints.experience_max is not None:
            min_val = constraints.experience_min if constraints.experience_min is not None else 0
            max_val = constraints.experience_max if constraints.experience_max is not None else 20
            constraints.allowed_experience_years = list(range(min_val, max_val + 1))

    # Log experience group details for debugging
    if constraints.num_employees and constraints.experience_groups:
        for group_name in ['Fresher', 'Junior', 'Mid', 'Senior', 'Expert']:
            grp = getattr(constraints.experience_groups, group_name, None)
            if grp and grp.min_ratio is not None:
                count = grp.min_ratio * constraints.num_employees
                print(f"🔍 {group_name} yêu cầu tối thiểu {count:.1f} người (~{round(count)} người)")
                
    return constraints, error_msg
