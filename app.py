from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from joblib import load
from lifelines import KaplanMeierFitter
import os
import SimpleITK as sitk
import tempfile
from radiomics import featureextractor
from scipy.integrate import simps
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Charger le modèle avec mise en cache
def load_model():
    model_path = 'coxph_model.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load(model_path)

# Seuil optimal pour séparer les groupes de risque
optimal_threshold = 3.038141178443309

# Charger les données pour tracer la courbe de Kaplan-Meier
def load_km_data():
    file_path = "km_curve_data.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    return df

# Fonction pour tracer les courbes de Kaplan-Meier
def plot_kaplan_meier(data):
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    for group in data['group'].unique():
        mask = data['group'] == group
        kmf.fit(data[mask]['TimeR'], event_observed=data[mask]['Rec'], label=group)
        survival_function = kmf.survival_function_
        fig.add_trace(go.Scatter(
            x=survival_function.index, 
            y=survival_function.iloc[:, 0],
            mode='lines',
            name=group
        ))
    fig.update_layout(
                      xaxis_title='Time (months)',
                      yaxis_title='Survival Probability',
                      width=500,  # Réduire la largeur de la figure
                      height=300)  # Réduire la hauteur de la figure
    
    return fig

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_cox = load_model()
    input_df = pd.DataFrame(data, index=[0])
    input_df['N'] = input_df['N'].astype('category')
    input_df['Thrombus'] = input_df['Thrombus'].astype('category')
    
    try:
        survival_function = model_cox.predict_survival_function(input_df)
        time_points = survival_function[0].x
        time_points = time_points[time_points <= 60]
        survival_probabilities = [fn(time_points) for fn in survival_function]
        survival_df = pd.DataFrame(survival_probabilities).transpose()
        survival_df.columns = ['Survival Probability']
        km_data = load_km_data()
        fig = plot_kaplan_meier(km_data)
        fig.add_trace(go.Scatter(x=time_points, y=survival_df['Survival Probability'], mode='lines', name='Patient-specific prediction', line=dict(color='blue', dash='dot')))
        fig.update_layout(xaxis_title='Time (months)', yaxis_title='Survival Probability')
        
        risk_score = model_cox.predict(input_df)[0]
        risk_group = "High risk" if risk_score >= optimal_threshold else "Low risk"
        
        return jsonify({
            'risk_group': risk_group,
            'figure': fig.to_json()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Endpoint pour générer des scores radiomiques
@app.route('/generate_radiomic_score', methods=['POST'])
def generate_radiomic_score():
    uploaded_ct = request.files['ct']
    uploaded_seg = request.files['seg']

    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_ct:
        tmp_ct.write(uploaded_ct.read())
        tmp_ct.seek(0)
        ct_image = sitk.ReadImage(tmp_ct.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as tmp_seg:
        tmp_seg.write(uploaded_seg.read())
        tmp_seg.seek(0)
        seg_image = sitk.ReadImage(tmp_seg.name)

    try:
        extractor = setup_extractor()
        feature_extraction_result = extractor.execute(ct_image, seg_image)
        features_df = pd.DataFrame([feature_extraction_result])
        
        features_of_interest = [
            'original_firstorder_10Percentile', 'original_firstorder_Mean', 'original_firstorder_Uniformity', 
            'original_glcm_ClusterTendency', 'original_glcm_Idm', 'original_glcm_Imc2', 'original_glcm_JointEnergy',
            'original_gldm_LargeDependenceEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 
            'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 
            'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 
            'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 
            'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_ZoneVariance', 
            'wavelet-LLH_firstorder_Entropy', 'wavelet-LLH_firstorder_InterquartileRange', 'wavelet-LLH_firstorder_Kurtosis', 
            'wavelet-LLH_glcm_Contrast', 'wavelet-LLH_glcm_DifferenceVariance', 'wavelet-LLH_glcm_Idm', 'wavelet-LLH_glcm_Idn', 
            'wavelet-LLH_glcm_Imc1', 'wavelet-LLH_gldm_HighGrayLevelEmphasis', 'wavelet-LLH_gldm_LargeDependenceEmphasis', 
            'wavelet-LLH_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LLH_glrlm_GrayLevelNonUniformityNormalized', 
            'wavelet-LLH_glrlm_HighGrayLevelRunEmphasis', 'wavelet-LLH_glrlm_LongRunLowGrayLevelEmphasis', 
            'wavelet-LLH_glrlm_RunLengthNonUniformity', 'wavelet-LLH_glrlm_RunPercentage', 'wavelet-LLH_ngtdm_Busyness', 
            'wavelet-LHL_firstorder_RobustMeanAbsoluteDeviation', 'wavelet-LHL_glcm_ClusterTendency', 'wavelet-LHL_glcm_Correlation', 
            'wavelet-LHL_glcm_DifferenceEntropy', 'wavelet-LHL_glcm_Idmn', 'wavelet-LHL_glcm_JointEntropy', 'wavelet-LHL_glcm_SumAverage', 
            'wavelet-LHL_gldm_DependenceNonUniformityNormalized', 'wavelet-LHL_glrlm_LongRunEmphasis', 'wavelet-LHL_glszm_SizeZoneNonUniformityNormalized', 
            'wavelet-LHL_ngtdm_Complexity', 'wavelet-LHH_firstorder_RootMeanSquared'
        ]

        selected_features_df = features_df[features_of_interest]

        rsf_model = load_model()
        scaler = load('scaler.joblib')

        time_points = np.linspace(0, 60, 61)
        cumulative_hazards = rsf_model.predict_cumulative_hazard_function(selected_features_df)
        rad_scores = np.array([simps([ch(tp) for tp in time_points], time_points) for ch in cumulative_hazards])
        normalized_rad_scores = scaler.transform(rad_scores.reshape(-1, 1)).flatten()

        return jsonify({
            'normalized_rad_score': normalized_rad_scores[0]
        })
    except Exception as e:
        return jsonify({'error': str(e)})

def setup_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings['normalize'] = True
    extractor.settings['normalizeScale'] = 100
    extractor.settings['resampledPixelSpacing'] = [1, 1, 1]
    extractor.settings['interpolator'] = sitk.sitkBSpline
    extractor.settings['binWidth'] = 25
    extractor.enableImageTypeByName('Wavelet')
    extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0, 4.0, 5.0]})
    return extractor

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
