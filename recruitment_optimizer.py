"""
Recruitment Resource Allocation Optimizer (DSS - Decision Support System)
------------------------------------------------------------------------
This script implements a Decision Support System that combines Machine Learning (ML)
and Integer Linear Programming (ILP) to optimize recruitment resource allocation.

Requirements:
    pip install pandas numpy pulp scikit-learn joblib

Usage:
    python recruitment_optimization_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import pulp

# ==========================================
# 1. EXPERIENCE LEVEL MAPPING FUNCTION
# ==========================================
def map_experience_level(years):
    """
    Maps experience years to standard candidate seniority levels.
    """
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

# ==========================================
# 2. INTEGER LINEAR PROGRAMMING (ILP) SOLVER
# ==========================================
def solve_recruitment_ilp(
    df,
    budget,
    headcount,
    allowed_titles=None,
    min_exp=0,
    max_exp=None,
    allowed_experience_years=None,
    min_skills=1,
    min_certs=0,
    min_avg_quality=0.0,
    job_title_bounds=None,      # dict: {job_title: (min_headcount, max_headcount)}
    job_title_max_ratios=None,  # dict: {job_title: max_ratio}
    title_balance_limit=None,   # Float: Max ratio of any single job title in recruited headcount (0.1 - 1.0)
    exp_requirements=None,      # dict: {level_name: {'min_ratio', 'max_ratio', 'exact_count', 'min_count', 'max_count'}}
    education_levels=None,      # list: allowed education levels (E_allow - Mục 5.5.1)
    edu_requirements=None,      # dict: {'levels': list, 'min_ratio': float, 'max_ratio': float}
    skills_high=None,           # dict: {'threshold': int, 'min_ratio': float, 'max_ratio': float}
    certifications_high=None,   # dict: {'threshold': int, 'min_ratio': float, 'max_ratio': float}
    remote_types=None,          # list: allowed remote work types (R_allow - Mục 5.8.1)
    remote_requirements=None    # dict: {remote_type: {'min_ratio': float, 'max_ratio': float}} or old format
):
    """
    Solves the recruitment resource allocation problem using Integer Linear Programming (ILP).
    Finds the optimal set of N candidates that maximizes the total quality score under budget B.
    """
    
    # 2.1. Hard Criteria Candidate Filtering
    filtered_df = df.copy()
    if allowed_titles:
        filtered_df = filtered_df[filtered_df['job_title'].isin(allowed_titles)]
        
    if allowed_experience_years is not None:
        filtered_df = filtered_df[filtered_df['experience_years'].isin(allowed_experience_years)]
        filtered_df = filtered_df[
            (filtered_df['skills_count'] >= min_skills) &
            (filtered_df['certifications'] >= min_certs)
        ]
    else:
        filtered_df = filtered_df[
            (filtered_df['experience_years'] >= min_exp) &
            (filtered_df['skills_count'] >= min_skills) &
            (filtered_df['certifications'] >= min_certs)
        ]
        if max_exp is not None:
            filtered_df = filtered_df[filtered_df['experience_years'] <= max_exp]
        
    # Hard filter for education_levels if provided (x_h = 0 for e not in E_allow)
    if education_levels:
        filtered_df = filtered_df[filtered_df['education_level'].isin(education_levels)]
            
    # Hard filter for remote_types if provided (x_h = 0 for r not in R_allow)
    remote_map = {'Remote': 'Yes', 'On-site': 'No', 'Hybrid': 'Hybrid'}
    if remote_types:
        db_remote_types = [remote_map[r] for r in remote_types if r in remote_map]
        if db_remote_types:
            filtered_df = filtered_df[filtered_df['remote_work'].isin(db_remote_types)]
    
    # Map experience levels using updated thresholds
    filtered_df['experience_level'] = filtered_df['experience_years'].apply(map_experience_level)
    
    # Reset index for PuLP variable mapping
    filtered_df = filtered_df.reset_index(drop=True)
    n_candidates = len(filtered_df)
    
    print(f"[DSS INFO] Number of eligible candidates in pool: {n_candidates}")
    
    if n_candidates == 0:
        print("[DSS WARNING] No candidates satisfy the hard recruitment criteria!")
        return None
        
    # 2.2. Initialize PuLP ILP Maximization Problem
    prob = pulp.LpProblem("Recruitment_Optimization", pulp.LpMaximize)
    
    # Decision variables: x_h = 1 if candidate h is selected, 0 otherwise
    x = [pulp.LpVariable(f"x_{h}", cat='Binary') for h in range(n_candidates)]
    
    # 2.3. Objective Function: Maximize total team quality score
    prob += pulp.lpSum(filtered_df.loc[h, 'quality_score'] * x[h] for h in range(n_candidates)), "Maximize_Team_Quality"
    
    # 2.4. Basic Constraints: Budget & Total Headcount
    # Total recruitment cost <= Budget
    prob += pulp.lpSum(filtered_df.loc[h, 'recruitment_cost'] * x[h] for h in range(n_candidates)) <= budget, "Budget_Limit"
    
    # Total selected candidates = Headcount
    prob += pulp.lpSum(x[h] for h in range(n_candidates)) == headcount, "Headcount_Limit"
    
    # 2.5. Job Title Allocation Bounds (job_title_limits)
    if job_title_bounds:
        for title, bounds in job_title_bounds.items():
            title_indices = filtered_df[filtered_df['job_title'] == title].index
            if len(title_indices) > 0:
                l_bound, u_bound = bounds
                if l_bound is not None:
                    prob += pulp.lpSum(x[h] for h in title_indices) >= l_bound, f"Min_Limit_{title.replace(' ', '_')}"
                if u_bound is not None:
                    prob += pulp.lpSum(x[h] for h in title_indices) <= u_bound, f"Max_Limit_{title.replace(' ', '_')}"
                    
    # Job Title Balance Constraint (Prevent monopolization by a single role)
    if title_balance_limit is not None:
        for title in filtered_df['job_title'].unique():
            title_indices = filtered_df[filtered_df['job_title'] == title].index
            prob += pulp.lpSum(x[h] for h in title_indices) <= title_balance_limit * headcount, f"Title_Balance_{title.replace(' ', '_')}"
            
    # Individual Job Title Max Ratio (job_title_max_ratio)
    if job_title_max_ratios:
        for title, max_ratio in job_title_max_ratios.items():
            if max_ratio is not None:
                title_indices = filtered_df[filtered_df['job_title'] == title].index
                prob += pulp.lpSum(x[h] for h in title_indices) <= max_ratio * headcount, f"Max_Ratio_Title_{title.replace(' ', '_')}"
            
    # 2.6. Minimum Team Average Quality constraint
    if min_avg_quality > 0:
        prob += pulp.lpSum(filtered_df.loc[h, 'quality_score'] * x[h] for h in range(n_candidates)) >= min_avg_quality * headcount, "Min_Average_Quality"
        
    # 2.7. Seniority Category Ratio & Headcount Constraints (experience_groups)
    if exp_requirements:
        for lvl, rules in exp_requirements.items():
            lvl_indices = filtered_df[filtered_df['experience_level'] == lvl].index
            if 'min_ratio' in rules and rules['min_ratio'] is not None:
                prob += pulp.lpSum(x[h] for h in lvl_indices) >= rules['min_ratio'] * headcount, f"Min_Ratio_{lvl}"
            if 'max_ratio' in rules and rules['max_ratio'] is not None:
                prob += pulp.lpSum(x[h] for h in lvl_indices) <= rules['max_ratio'] * headcount, f"Max_Ratio_{lvl}"
            if 'exact_count' in rules and rules['exact_count'] is not None:
                prob += pulp.lpSum(x[h] for h in lvl_indices) == rules['exact_count'], f"Exact_Count_{lvl}"
            if 'min_count' in rules and rules['min_count'] is not None:
                prob += pulp.lpSum(x[h] for h in lvl_indices) >= rules['min_count'], f"Min_Count_{lvl}"
            if 'max_count' in rules and rules['max_count'] is not None:
                prob += pulp.lpSum(x[h] for h in lvl_indices) <= rules['max_count'], f"Max_Count_{lvl}"
                
    # 2.8. Education Level Ratio Constraints (education_ratio_min/max)
    if edu_requirements:
        edu_lvls = edu_requirements.get('levels', [])
        if not edu_lvls and education_levels:
            edu_lvls = education_levels
            
        if edu_lvls:
            edu_indices = filtered_df[filtered_df['education_level'].isin(edu_lvls)].index
            if 'min_ratio' in edu_requirements and edu_requirements['min_ratio'] is not None:
                prob += pulp.lpSum(x[h] for h in edu_indices) >= edu_requirements['min_ratio'] * headcount, "Min_Education_Level_Ratio"
            if 'max_ratio' in edu_requirements and edu_requirements['max_ratio'] is not None:
                prob += pulp.lpSum(x[h] for h in edu_indices) <= edu_requirements['max_ratio'] * headcount, "Max_Education_Level_Ratio"
                
    # 2.9. High Skills Constraints (skills_high)
    if skills_high and 'threshold' in skills_high:
        thresh = skills_high['threshold']
        skills_high_indices = filtered_df[filtered_df['skills_count'] >= thresh].index
        if 'min_ratio' in skills_high and skills_high['min_ratio'] is not None:
            prob += pulp.lpSum(x[h] for h in skills_high_indices) >= skills_high['min_ratio'] * headcount, "Min_Skills_High_Ratio"
        if 'max_ratio' in skills_high and skills_high['max_ratio'] is not None:
            prob += pulp.lpSum(x[h] for h in skills_high_indices) <= skills_high['max_ratio'] * headcount, "Max_Skills_High_Ratio"
            
    # 2.10. High Certifications Constraints (certifications_high)
    if certifications_high and 'threshold' in certifications_high:
        thresh = certifications_high['threshold']
        certs_high_indices = filtered_df[filtered_df['certifications'] >= thresh].index
        if 'min_ratio' in certifications_high and certifications_high['min_ratio'] is not None:
            prob += pulp.lpSum(x[h] for h in certs_high_indices) >= certifications_high['min_ratio'] * headcount, "Min_Certs_High_Ratio"
        if 'max_ratio' in certifications_high and certifications_high['max_ratio'] is not None:
            prob += pulp.lpSum(x[h] for h in certs_high_indices) <= certifications_high['max_ratio'] * headcount, "Max_Certs_High_Ratio"
        
    # 2.11. Work Mode (Hybrid/Remote) Constraints
    # New format remote_requirements dict mapping remote_type to ratios
    if remote_requirements and remote_types:
        for rtype in remote_types:
            db_type = remote_map.get(rtype)
            if db_type and rtype in remote_requirements:
                remote_indices = filtered_df[filtered_df['remote_work'] == db_type].index
                rules = remote_requirements[rtype]
                if 'min_ratio' in rules and rules['min_ratio'] is not None:
                    prob += pulp.lpSum(x[h] for h in remote_indices) >= rules['min_ratio'] * headcount, f"Min_Ratio_{rtype}"
                if 'max_ratio' in rules and rules['max_ratio'] is not None:
                    prob += pulp.lpSum(x[h] for h in remote_indices) <= rules['max_ratio'] * headcount, f"Max_Ratio_{rtype}"
                    
    # Old format backward compatibility
    elif remote_requirements:
        if 'min_hybrid_ratio' in remote_requirements:
            hybrid_indices = filtered_df[filtered_df['remote_work'] == 'Hybrid'].index
            prob += pulp.lpSum(x[h] for h in hybrid_indices) >= remote_requirements['min_hybrid_ratio'] * headcount, "Min_Hybrid_Ratio"
        if 'max_remote_ratio' in remote_requirements:
            remote_indices = filtered_df[filtered_df['remote_work'] == 'Yes'].index
            prob += pulp.lpSum(x[h] for h in remote_indices) <= remote_requirements['max_remote_ratio'] * headcount, "Max_Remote_Ratio"
            
    # 2.12. Solve Optimization Problem
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    
    if pulp.LpStatus[status] == 'Optimal':
        selected_indices = [h for h in range(n_candidates) if pulp.value(x[h]) > 0.5]
        selected_candidates = filtered_df.iloc[selected_indices].copy()
        print("[DSS SUCCESS] Optimal recruitment plan found.")
        return selected_candidates
    else:
        print(f"[DSS ERROR] Could not find a feasible solution. Solver status: {pulp.LpStatus[status]}")
        return None

# ==========================================
# 3. DEMO WORKFLOW & SENSITIVITY TESTING
# ==========================================
if __name__ == "__main__":
    # Check paths
    csv_path = os.path.join("data", "job_salary_prediction_dataset.csv")
    model_path = "best_salary_prediction_model.pkl"
    
    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        print(f"Error: Missing dataset ({csv_path}) or model ({model_path})!")
        print("Please ensure you are running the script in the directory containing these files.")
        exit(1)
        
    print("Loading candidate dataset...")
    salary_df = pd.read_csv(csv_path)
    
    print("Loading ML model...")
    best_model = joblib.load(model_path)
    
    print("Calculating quality scores and predicted salaries...")
    # Composite Quality Index calculation
    s = 10 * (salary_df['skills_count'] - 1) / 18
    t = 10 * salary_df['certifications'] / 5
    y = 10 * np.log1p(salary_df['experience_years']) / np.log(21)
    
    edu_mapping = {'High School': 2, 'Diploma': 4, 'Bachelor': 6, 'Master': 8, 'PhD': 10}
    d = salary_df['education_level'].map(edu_mapping)
    
    company_mapping = {'Startup': 4, 'Small': 5, 'Medium': 6, 'Large': 8, 'Enterprise': 10}
    f = salary_df['company_size'].map(company_mapping)
    
    # q = 0.25*s + 0.08*t + 0.47*y + 0.11*d + 0.09*f
    salary_df['quality_score'] = 0.25 * s + 0.08 * t + 0.47 * y + 0.11 * d + 0.09 * f
    
    # Predict cost using ML: cost = 1.3 * predicted_salary
    X_features = salary_df.drop('salary', axis=1)
    salary_df['predicted_salary'] = best_model.predict(X_features)
    salary_df['recruitment_cost'] = 1.3 * salary_df['predicted_salary']
    
    # RUN SAMPLE SCENARIO
    print("\n--- Running Sample Optimization Scenario ---")
    budget_val = 1500000
    headcount_val = 10
    allowed_titles = ['AI Engineer', 'Data Scientist', 'Data Analyst']
    
    job_title_bounds = {
        'AI Engineer': (1, 4),
        'Data Scientist': (1, 4),
        'Data Analyst': (1, 4)
    }
    
    # Redefined experience levels mapping verification
    exp_req = {
        'Senior': {'min_ratio': 0.3},
        'Fresher': {'max_ratio': 0.3}
    }
    
    edu_req = {
        'levels': ['Master', 'PhD'],
        'min_ratio': 0.4
    }
    
    remote_req = {
        'min_hybrid_ratio': 0.5,
        'max_remote_ratio': 0.2
    }
    
    results = solve_recruitment_ilp(
        df=salary_df,
        budget=budget_val,
        headcount=headcount_val,
        allowed_titles=allowed_titles,
        min_exp=1,
        min_skills=8,
        min_certs=0,
        job_title_bounds=job_title_bounds,
        title_balance_limit=0.4,
        exp_requirements=exp_req,
        edu_requirements=edu_req,
        remote_requirements=remote_req
    )
    
    if results is not None:
        print("\n" + "="*45)
        print("                 OPTIMAL PLAN")
        print("="*45)
        print(f"Total Selected: {len(results)}")
        print(f"Actual Cost:    ${results['recruitment_cost'].sum():,.2f} (Budget: ${budget_val:,.2f})")
        print(f"Avg Quality:    {results['quality_score'].mean():.2f}/10.0")
        print("\nRecruited Canditates List:")
        print(results[['job_title', 'experience_years', 'experience_level', 'education_level', 'remote_work', 'recruitment_cost', 'quality_score']])
    else:
        print("Failed to solve.")
