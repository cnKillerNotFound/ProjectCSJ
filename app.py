# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from model_training import PlacementPredictor

# 设置页面配置
st.set_page_config(
    page_title="学生就业预测系统",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载模型
@st.cache_resource
def load_model():
    try:
        predictor = PlacementPredictor.load_model('placement_predictor.joblib')
        return predictor
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None

def main():
    # 标题
    st.title("🎓 学生就业预测系统")
    st.markdown("---")
    
    # 加载模型
    with st.spinner("正在加载预测模型..."):
        predictor = load_model()
    
    if predictor is None:
        st.error("无法加载预测模型，请确保模型文件存在")
        return
    
    # 创建两列布局
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 输入学生信息")
        
        # 创建表单
        with st.form("student_info_form"):
            # 第一行
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                iq = st.slider("智商分数 (IQ)", min_value=70, max_value=140, value=100, 
                              help="通常分布在100左右")
                prev_sem_result = st.slider("上学期平均绩点", min_value=5.0, max_value=10.0, 
                                          value=7.5, step=0.1,
                                          help="范围：5.0 至 10.0")
            
            with col1_2:
                cgpa = st.slider("累计平均绩点 (CGPA)", min_value=5.0, max_value=10.0, 
                               value=7.8, step=0.1,
                               help="范围：约5.0至10.0")
                academic_performance = st.slider("年度学术评分", min_value=1, max_value=10, 
                                              value=7,
                                              help="评分标准：1至10分")
            
            # 第二行
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                internship = st.radio("实习经验", options=["有", "无"], horizontal=True,
                                    help="学生是否已完成任何实习")
                extra_curricular = st.slider("课外活动参与度", min_value=0, max_value=10, 
                                           value=5,
                                           help="评分范围为0至10分")
            
            with col2_2:
                communication_skills = st.slider("软技能评分", min_value=1, max_value=10, 
                                               value=6,
                                               help="评分范围：1至10")
                projects_completed = st.slider("已完成项目数量", min_value=0, max_value=5, 
                                             value=2,
                                             help="已完成的学术/技术项目数量（0至5个）")
            
            # 提交按钮
            submitted = st.form_submit_button("预测就业概率", use_container_width=True)
    
    with col2:
        st.header("📈 预测结果")
        
        if submitted:
            # 准备输入数据
            input_data = {
                'IQ': iq,
                'Prev_Sem_Result': prev_sem_result,
                'CGPA': cgpa,
                'Academic_Performance': academic_performance,
                'Internship_Experience': internship == "有",
                'Extra_Curricular_Score': extra_curricular,
                'Communication_Skills': communication_skills,
                'Projects_Completed': projects_completed
            }
            
            # 进行预测
            with st.spinner("正在分析..."):
                result = predictor.predict(input_data)
            
            # 显示结果
            probability = result['probability']
            prediction = result['prediction']
            
            # 显示概率进度条
            st.subheader("就业概率")
            st.progress(probability)
            st.metric("概率值", f"{probability:.2%}")
            
            # 显示预测结果
            if probability >= 0.7:
                st.success(f"**预测结果: {prediction}** 🎉")
                st.balloons()
            elif probability >= 0.5:
                st.warning(f"**预测结果: {prediction}** ⚠️")
            else:
                st.error(f"**预测结果: {prediction}** 💡")
            
            # 显示详细分析
            with st.expander("详细分析"):
                st.write(f"- **就业概率**: {probability:.2%}")
                st.write(f"- **预测类别**: {prediction}")
                st.write(f"- **模型置信度**: {'高' if probability > 0.7 or probability < 0.3 else '中等'}")
                
                # 给出建议
                st.subheader("💡 改进建议")
                if probability < 0.5:
                    if internship == "无":
                        st.write("- ✅ 考虑参加实习项目")
                    if projects_completed < 3:
                        st.write("- ✅ 增加项目经验")
                    if communication_skills < 7:
                        st.write("- ✅ 提升沟通技巧")
                    if extra_curricular < 5:
                        st.write("- ✅ 参与更多课外活动")
                else:
                    st.write("- 🎉 保持当前良好表现!")
        
        else:
            st.info("请在左侧输入学生信息并点击'预测就业概率'")
    
    # 添加模型信息部分
    st.markdown("---")
    st.header("ℹ️ 模型信息")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("模型性能")
        st.write(f"- **测试准确率**: {predictor.model_info['test_accuracy']:.4f}")
        st.write(f"- **AUC分数**: {predictor.model_info['test_auc']:.4f}")
        st.write(f"- **最佳参数**: {predictor.model_info['best_params']}")
    
    with col4:
        st.subheader("特征说明")
        features_info = {
            "IQ": "学生的智商分数",
            "Prev_Sem_Result": "上个学期的平均绩点",
            "CGPA": "累计平均绩点",
            "Academic_Performance": "年度学术评分",
            "Internship_Experience": "是否已完成实习",
            "Extra_Curricular_Score": "课外活动参与度",
            "Communication_Skills": "软技能评分",
            "Projects_Completed": "已完成的学术/技术项目数量"
        }
        
        for feature, desc in features_info.items():
            st.write(f"- **{feature}**: {desc}")

if __name__ == "__main__":
    main()