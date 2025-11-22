# model_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('大学生就业去向影响因素数据集.csv', encoding='Windows-1252')

class PlacementPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_info = {}
        
    def load_and_preprocess_data(self, df):
        """加载和预处理数据"""
        print("开始数据预处理...")
        
        # 复制数据避免修改原数据
        data = df.copy()
        
        # 检查缺失值
        print(f"缺失值统计:\n{data.isnull().sum()}")
        
        # 处理缺失值 - 用中位数填充数值列
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # 处理分类列
        if 'Internship_Experience' in data.columns:
            data['Internship_Experience'] = data['Internship_Experience'].map({True: 1, False: 0})
        
        # 编码目标变量
        if 'Placement' in data.columns:
            data['Placement'] = data['Placement'].map({True: 1, False: 0})
        
        print("数据预处理完成!")
        return data
    
    def feature_engineering(self, data):
        """特征工程"""
        print("开始特征工程...")
        
        # 创建新特征
        data['Academic_Stability'] = data['CGPA'] - data['Prev_Sem_Result']
        data['Overall_Performance'] = (data['CGPA'] + data['Academic_Performance']) / 2
        data['Skill_Composite'] = (data['Communication_Skills'] + data['Extra_Curricular_Score']) / 2
        data['Productivity_Score'] = data['Projects_Completed'] * data['Academic_Performance']
        
        # 交互特征
        data['IQ_Academic_Interaction'] = data['IQ'] * data['CGPA']
        data['Internship_Projects'] = data['Internship_Experience'] * data['Projects_Completed']
        
        print(f"特征工程完成! 新增特征: {list(data.columns[-6:])}")
        return data
    
    def prepare_features(self, data):
        """准备特征和目标变量"""
        # 排除非特征列
        exclude_cols = ['College_ID', 'Placement']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols]
        y = data['Placement'] if 'Placement' in data.columns else None
        
        print(f"特征数量: {len(feature_cols)}")
        print(f"特征列表: {feature_cols}")
        
        if y is not None:
            print(f"目标变量分布:\n{y.value_counts()}")
            print(f"就业率: {y.mean():.2%}")
        
        return X, y, feature_cols
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """训练模型"""
        print("\n开始训练模型...")
        
        # 创建管道
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
        ])
        
        # 超参数网格
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        }
        
        # 网格搜索
        grid_search = GridSearchCV(
            pipeline, param_grid, 
            cv=5, scoring='roc_auc', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 存储最佳模型
        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score_cv = grid_search.best_score_
        
        # 在测试集上评估
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        test_accuracy = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"模型训练完成!")
        print(f"交叉验证 AUC: {best_score_cv:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")
        print(f"测试集 AUC: {test_auc:.4f}")
        print(f"最佳参数: {best_params}")
        
        # 存储模型信息
        self.model_info = {
            'test_accuracy': test_accuracy,
            'test_auc': test_auc,
            'best_params': best_params,
            'feature_names': X_train.columns.tolist()
        }
        
        return test_accuracy, test_auc
    
    def save_model(self, filepath='placement_predictor.joblib'):
        """保存模型和相关信息"""
        if self.model is None:
            print("没有训练好的模型可以保存!")
            return
        
        # 保存模型
        joblib.dump(self.model, filepath)
        
        # 保存模型信息
        info_filepath = filepath.replace('.joblib', '_info.json')
        with open(info_filepath, 'w') as f:
            json.dump(self.model_info, f, indent=4)
        
        print(f"模型已保存到: {filepath}")
        print(f"模型信息已保存到: {info_filepath}")
    
    @classmethod
    def load_model(cls, filepath='placement_predictor.joblib'):
        """加载模型和相关信息"""
        # 加载模型
        model = joblib.load(filepath)
        
        # 加载模型信息
        info_filepath = filepath.replace('.joblib', '_info.json')
        with open(info_filepath, 'r') as f:
            model_info = json.load(f)
        
        # 创建预测器实例
        predictor = cls()
        predictor.model = model
        predictor.model_info = model_info
        predictor.feature_names = model_info['feature_names']
        
        print(f"模型已从 {filepath} 加载")
        print(f"模型准确率: {model_info['test_accuracy']:.4f}")
        print(f"模型AUC: {model_info['test_auc']:.4f}")
        
        return predictor
    
    def predict(self, input_data):
        """预测就业概率"""
        if self.model is None:
            print("请先加载或训练模型!")
            return None
        
        # 确保输入数据是DataFrame
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data])
        
        # 特征工程
        input_data = self._apply_feature_engineering(input_data)
        
        # 确保有所有必要的特征
        input_data = self._ensure_features(input_data)
        
        # 预测
        probability = self.model.predict_proba(input_data)[0, 1]
        prediction = self.model.predict(input_data)[0]
        
        return {
            'probability': probability,
            'prediction': '高概率就业' if prediction == 1 else '低概率就业',
            'prediction_bool': prediction == 1
        }
    
    def _apply_feature_engineering(self, data):
        """应用特征工程"""
        # 创建新特征
        data['Academic_Stability'] = data['CGPA'] - data['Prev_Sem_Result']
        data['Overall_Performance'] = (data['CGPA'] + data['Academic_Performance']) / 2
        data['Skill_Composite'] = (data['Communication_Skills'] + data['Extra_Curricular_Score']) / 2
        data['Productivity_Score'] = data['Projects_Completed'] * data['Academic_Performance']
        data['IQ_Academic_Interaction'] = data['IQ'] * data['CGPA']
        
        # 处理Internship_Experience
        if 'Internship_Experience' in data.columns:
            if data['Internship_Experience'].dtype == 'bool':
                data['Internship_Experience'] = data['Internship_Experience'].astype(int)
            data['Internship_Projects'] = data['Internship_Experience'] * data['Projects_Completed']
        
        return data
    
    def _ensure_features(self, data):
        """确保数据包含所有必要的特征"""
        for feature in self.feature_names:
            if feature not in data.columns:
                print(f"警告: 特征 '{feature}' 不在输入数据中，将用0填充")
                data[feature] = 0
        
        # 重新排列列的顺序以匹配训练数据
        data = data[self.feature_names]
        
        return data

def train_and_save_model():
    """训练并保存模型"""
    print("=== 学生就业预测模型训练 ===")
    
    # 初始化预测器
    predictor = PlacementPredictor()
    
    print(f"数据集大小: {df.shape}")
    
    # 数据预处理
    processed_data = predictor.load_and_preprocess_data(df)
    
    # 特征工程
    engineered_data = predictor.feature_engineering(processed_data)
    
    # 准备特征和目标变量
    X, y, feature_cols = predictor.prepare_features(engineered_data)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 训练模型
    accuracy, auc_score = predictor.train_model(X_train, X_test, y_train, y_test)
    
    # 保存模型
    predictor.save_model('placement_predictor.joblib')
    
    print("\n=== 训练完成 ===")
    print(f"最终测试准确率: {accuracy:.4f}")
    print(f"最终AUC分数: {auc_score:.4f}")
    
    return predictor

if __name__ == "__main__":
    # 训练并保存模型
    trained_predictor = train_and_save_model()