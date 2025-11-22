# predict.py
import pandas as pd
from model_training import PlacementPredictor

def predict_single_student():
    """预测单个学生的就业概率"""
    # 加载模型
    predictor = PlacementPredictor.load_model('placement_predictor.joblib')
    
    # 输入学生信息
    print("请输入学生信息:")
    iq = int(input("智商分数 (IQ): "))
    prev_sem_result = float(input("上学期平均绩点: "))
    cgpa = float(input("累计平均绩点 (CGPA): "))
    academic_performance = int(input("年度学术评分 (1-10): "))
    internship = input("是否有实习经验 (是/否): ").strip().lower() == '是'
    extra_curricular = int(input("课外活动参与度 (0-10): "))
    communication_skills = int(input("软技能评分 (1-10): "))
    projects_completed = int(input("已完成项目数量 (0-5): "))
    
    # 准备输入数据
    input_data = {
        'IQ': iq,
        'Prev_Sem_Result': prev_sem_result,
        'CGPA': cgpa,
        'Academic_Performance': academic_performance,
        'Internship_Experience': internship,
        'Extra_Curricular_Score': extra_curricular,
        'Communication_Skills': communication_skills,
        'Projects_Completed': projects_completed
    }
    
    # 进行预测
    result = predictor.predict(input_data)
    
    # 显示结果
    print("\n" + "="*50)
    print("预测结果:")
    print(f"就业概率: {result['probability']:.2%}")
    print(f"预测类别: {result['prediction']}")
    print("="*50)

def predict_batch_students(csv_file):
    """批量预测学生的就业概率"""
    # 加载模型
    predictor = PlacementPredictor.load_model('placement_predictor.joblib')
    
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 进行预测
    results = []
    for _, row in df.iterrows():
        result = predictor.predict(row.to_dict())
        results.append({
            'College_ID': row.get('College_ID', 'N/A'),
            '就业概率': result['probability'],
            '预测结果': result['prediction']
        })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    output_file = csv_file.replace('.csv', '_predictions.csv')
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"批量预测完成! 结果已保存到: {output_file}")
    return results_df

if __name__ == "__main__":
    # 选择预测模式
    print("选择预测模式:")
    print("1. 单个学生预测")
    print("2. 批量学生预测")
    
    choice = input("请输入选择 (1/2): ").strip()
    
    if choice == '1':
        predict_single_student()
    elif choice == '2':
        csv_file = input("请输入CSV文件路径: ").strip()
        predict_batch_students(csv_file)
    else:
        print("无效选择!")