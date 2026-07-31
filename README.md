# Hướng Dẫn Kỹ Thuật: Hệ Thống Hỗ Trợ Ra Quyết Định Phân Bổ Nguồn Lực Tuyển Dụng (DSS - Recruitment ILP)

Tài liệu này giải thích chi tiết cấu trúc mô hình kết hợp Machine Learning và Quy hoạch tuyến tính nguyên (Integer Linear Programming - ILP) để tối ưu cơ cấu nhân sự.

---

## 1. Bài Toán Tối Ưu Hóa (Problem Formulation)

Mục tiêu là tìm ra tổ hợp tuyển dụng gồm đúng $N$ nhân sự thỏa mãn giới hạn ngân sách $B$, sao cho **tổng chất lượng chuyên môn của đội ngũ mới là lớn nhất**.

### Biến Quyết Định (Decision Variables)
Với mỗi ứng viên $h$ trong tập dữ liệu (sau khi lọc các tiêu chuẩn cứng):
$$x_h = \begin{cases} 
1 & \text{nếu tuyển ứng viên } h \\ 
0 & \text{ngược lại} 
\end{cases}$$

### Hàm Mục Tiêu (Objective Function)
Tối đa hóa tổng điểm chất lượng đội ngũ:
$$\text{Maximize } \sum_{h=1}^{M} q_h \cdot x_h$$
Trong đó:
* $M$ là số ứng viên thỏa mãn tiêu chuẩn cứng.
* $q_h$ là điểm chất lượng tổng hợp (Composite Quality Index) của ứng viên $h$.

---

## 2. Cách Tính Toán Các Chỉ Số Đầu Vào

### 2.1. Điểm Chất Lượng Tổng Hợp ($q_h$)
Điểm chất lượng của từng ứng viên được tính bằng trọng số của 5 thành phần:
$$q_h = 0.25 \cdot s_h + 0.08 \cdot t_h + 0.47 \cdot y_h + 0.11 \cdot d_h + 0.09 \cdot f_h$$

Trong đó các điểm thành phần được định chuẩn về thang điểm $10$:
1. **Điểm kỹ năng ($s_h$)**: Chuẩn hóa từ cột `skills_count` $k \in [1, 19]$:
   $$s_h = 10 \times \frac{k - 1}{18}$$
2. **Điểm chứng chỉ ($t_h$)**: Chuẩn hóa từ cột `certifications` $\ell \in [0, 5]$:
   $$t_h = 10 \times \frac{\ell}{5}$$
3. **Điểm kinh nghiệm ($y_h$)**: Áp dụng hàm logarit phi tuyến tính (vì số năm kinh nghiệm tăng cao ở mức senior/expert không đem lại hiệu quả tuyến tính tương xứng như junior):
   $$y_h = 10 \times \frac{\log(1 + j_h)}{\log(21)} \quad (\text{với } j_h \text{ là } \text{experience\_years} \in [0, 20])$$
4. **Điểm trình độ học vấn ($d_h$)**: Ánh xạ từ cột `education_level`:
   * High School: $2$ | Diploma: $4$ | Bachelor: $6$ | Master: $8$ | PhD: $10$
5. **Điểm quy mô công ty ($f_h$)**: Ánh xạ từ cột `company_size`:
   * Startup: $4$ | Small: $5$ | Medium: $6$ | Large: $8$ | Enterprise: $10$

### 2.2. Chi Phí Tuyển Dụng Dự Báo ($p_h$)
Chi phí thực tế để tuyển dụng ứng viên $h$ bao gồm lương và các chi phí phát sinh khác (onboarding, MacBook, phúc lợi, bảo hiểm...). Chi phí này được giả định bằng $1.3$ lần mức lương dự báo:
$$p_h = 1.3 \times \hat{s}_h$$
Trong đó $\hat{s}_h$ là mức lương dự báo bằng mô hình Machine Learning **HistGradientBoostingRegressor** (đã được lưu trong tệp `best_salary_prediction_model.pkl`).

---

## 3. Các Ràng Buộc Trong Mô Hình (Constraints)

1. **Ràng buộc ngân sách (Recruitment Budget)**:
   $$\sum_{h=1}^{M} p_h \cdot x_h \le B$$
2. **Ràng buộc số lượng (Headcount Limit)**:
   $$\sum_{h=1}^{M} x_h = N$$
3. **Ràng buộc cơ cấu chức danh**:
   $$L_i \le \sum_{h \in \text{Role } i} x_h \le U_i \quad \forall i \in I$$
4. **Ràng buộc cân bằng chức danh** (tỉ lệ trần $\delta$ để tránh dồn hết ngân sách vào một chức danh):
   $$\sum_{h \in \text{Role } i} x_h \le \delta \cdot N \quad \forall i \in I$$
5. **Ràng buộc phân loại kinh nghiệm** (Áp dụng ánh xạ số năm kinh nghiệm mới):
   * *Fresher*: $[0, 1)$ năm | *Junior*: $[1, 3)$ năm | *Mid*: $[3, 5)$ năm | *Senior*: $[5, 10)$ năm | *Expert*: $\ge 10$ năm
   * Mô hình thiết lập các tỉ lệ trần/sàn cho từng nhóm này, ví dụ:
     $$\sum_{h \in \text{Senior}} x_h \ge \rho_{\text{senior}} \cdot N$$
     $$\sum_{h \in \text{Fresher}} x_h \le \rho_{\text{fresher}} \cdot N$$
6. **Ràng buộc trình độ học vấn** (ví dụ: Tỉ lệ Master/PhD tối thiểu):
   $$\sum_{h \in \{\text{Master, PhD}\}} x_h \ge \rho_{\text{edu}} \cdot N$$
7. **Ràng buộc hình thức làm việc**:
   * Tỉ lệ Hybrid tối thiểu: $\sum_{h \in \text{Hybrid}} x_h \ge \rho_{\text{hybrid}} \cdot N$
   * Tỉ lệ Remote tối đa: $\sum_{h \in \text{Remote}} x_h \le \rho_{\text{remote}} \cdot N$

## 4. Hướng Dẫn Cài Đặt và Chạy Chương Trình

### 4.1. Yêu Cầu Hệ Thống & Cài Đặt Thư Viện

Đảm bảo hệ thống của bạn đã cài đặt Python (phiên bản 3.8 trở lên). Sau đó, chạy lệnh dưới đây trong terminal tại thư mục gốc của dự án để cài đặt tất cả các thư viện cần thiết:

```bash
pip install pandas numpy pulp scikit-learn joblib streamlit plotly pydantic google-generativeai
```

> [!NOTE]
> Hệ thống sử dụng bộ giải (solver) mặc định của PuLP là **CBC**. Bộ giải này thường đi kèm sẵn khi cài đặt thư viện `pulp` trên hầu hết hệ điều hành phổ biến.

### 4.2. Chạy Giao Diện Web Dashboard (`app.py`)

Ứng dụng web Streamlit cung cấp giao diện đồ họa cao cấp, hỗ trợ chế độ chatbot thông minh và các tab phân tích nâng cao (Độ nhạy, What-if, Monte Carlo). Chạy lệnh sau để khởi động:

```bash
streamlit run app.py
```

Trình duyệt web sẽ tự động mở ứng dụng tại địa chỉ cục bộ (mặc định: `http://localhost:8501`).

### 4.3. Chạy File CLI Độc Lập (`recruitment_optimizer.py`)

Tệp [recruitment_optimizer.py](./recruitment_optimizer.py) chứa mã nguồn giải bài toán ILP tối ưu hóa rút gọn, chạy trực tiếp trên giao diện dòng lệnh (CLI). Để kiểm tra nhanh thuật toán mà không cần giao diện web, chạy lệnh:

```bash
python recruitment_optimizer.py
```

Mô hình sẽ tự động tính toán điểm chất lượng, dự báo chi phí bằng ML, thiết lập các ràng buộc toán học bằng PuLP, và tìm ra danh sách 10 nhân sự tối ưu nhất thỏa mãn ngân sách \$1,500,000.
