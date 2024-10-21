import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib  

def preprocess_and_predict(csv_path, scaler_pkl, model_pkl):
    columns = ['FilterItLogS','C2SP2','SLogP','nBondsM','MIC5','MIC3','VE1_DzZ','BCUTp-1h',	'BCUTse-1l','ATS2Z','AATS0v','Mp','PEOE_VSA6','BertzCT','ETA_eta_FL','n5aHRing','piPC8','ZMIC1','GATS1m','MATS1are',	'ATSC1pe',	'BCUTZ-1l',	'SpMax_D',	'AATS0p',	'ATS0Z',	'ETA_beta',	'BCUTm-1l',	'ATSC0m',	'GATS1v',	'MIC4',	'LogEE_D',	'MW',	'SZ',	'ETA_beta_ns',	'ATS0m',	'VE1_D',	'Sm',	'AATS1p',	'ZMIC0',	'TSRW10',	'VE2_Dzi',	'VE2_Dzv',	'VE1_Dzare',	'ATSC3m',	'ETA_alpha',	'GATS3Z',	'n5aRing',	'MATS1pe',	'AATSC1are',	'ATS6Z',	'fMF',	'HybRatio',	'VE1_Dzp',	'SRW09',	'nBondsKD',	'piPC4',	'AATSC0dv',	'SIC5',	'AATS4m',	'ZMIC2',	'AMW',	'VE3_DzZ',	'AETA_alpha',	'AATS0Z',	'Diameter',	'nAcid',	'ATSC2are',	'Xp-3dv',	'AATSC1pe',	'AATS5d',	'piPC6',	'n5Ring',	'BCUTpe-1l',	'TIC0',	'ETA_dEpsilon_D',	'ETA_epsilon_5',	'VE3_Dzse',	'BCUTv-1h',	'SsCl',	'AATS4v',	'Mv',	'GATS3m',	'AATS0m',	'GATS1pe',	'SlogP_VSA2',	'TpiPC10',	'VE3_Dzpe',	'ATS5s',	'AETA_eta_FL',	'VE1_Dzpe',	'GATS2d',	'ATSC1c',	'MID_h',	'VR3_D',	'BCUTd-1h',	'ATSC1se',	'MATS2i',	'AATSC1c',	'AATS1m',	'nBondsA',	'ATSC1s',	'LogEE_Dzv',	'MZ',	'MATS3m',	'AATSC0Z',	'VR2_D',	'GATS1i',	'AATSC0v',	'AXp-4d',	'NsOH',	'MATS3Z',	'AETA_eta_BR',	'ETA_psi_1',	'piPC9',	'AATSC1m',	'nAromBond',	'AATS2m',	'ETA_dEpsilon_B',	'GATS1Z',	'AXp-2dv',	'SpMax_Dzv',	'AATS4p',	'VE3_Dzare',	'SMR_VSA7',	'AATSC2i',	'ATS8Z',	'AATS1v',	'piPC3',	'GATS1s',	'Xch-6dv',	'AATSC1s',	'GATS4i',	'ATSC3Z',	'TMWC10',	'AATSC2s',	'SpAbs_D',	'SIC3',	'AATSC3se',	'AATS5p',	'VR2_Dzare',	'MATS1Z',	'ATSC4pe',	'ATS5se',	'ETA_epsilon_2',	'n10FaHRing',	'nCl',	'ATSC1Z',	'VE2_Dzp',	'AATS4Z',	'ATS7m',	'TopoPSA(NO)',	'MATS3s',	'GATS2s',	'BCUTi-1l',	'VSA_EState6',	'nHRing',	'Sp',	'AATS2Z',	'SsOH',	'piPC2',	'ATS1p',	'BIC3',	'AATSC0p',	'PEOE_VSA2',	'FCSP3',	'MATS4Z',	'BCUTare-1l',	'SpMAD_Dzp',	'MIC2',	'AATSC5c',	'Xp-5dv',	'MATS1v',	'AXp-3dv',	'BCUTd-1l',	'ATSC1are',	'BCUTm-1h',	'VE1_Dzv',	'MID_C',	'ZMIC4',	'Xp-0d',	'GATS1d',	'WPath',	'VSA_EState1',	'ATSC4m',	'NtsC',	'AATSC1v',	'GATS1p',	'MATS4m',	'GATS5i',	'MATS2s',	'Mi',	'AXp-5d',	'SRW05',	'BIC0',	'VR2_A',	'Xp-1dv',	'AATS3se',	'Xch-5d',	'NsssCH',	'GATS4m',	'ATSC3s',	'Xch-5dv',	'MATS5i',	'ATS5m',	'GATS1se',	'ATS5pe',	'ATSC5i',	'Xch-6d',	'ABCGG',	'ATSC5pe',	'SpAD_Dzp',	'AATSC4dv',	'CIC2',	'SMR_VSA2',	'ATSC3p',	'AATS2s',	'ATSC3d',	'LabuteASA',	'Xc-3dv',	'GATS1are',	'VR3_DzZ',	'StN',	'VR1_Dzp',	'ETA_dEpsilon_C',	'ATS4are',	'AATS2pe',	'ATS3p',	'AATS4are',	'MATS3dv',	'VR3_Dzp',	'VR2_Dzse',	'CIC4',	'AETA_beta_s',	'ATSC5m',	'ATSC8i',	'AATSC3m',	'ATSC1v',	'VE1_Dzse',	'AATSC2Z',	'ATS4m',	'AATSC0m',	'AATS1Z',	'ATS4s',	'ATS7Z',	'AATSC0se',	'Xpc-5d',	'AATSC0pe',	'VE2_DzZ',	'ATSC2se',	'SIC0',	'ETA_shape_x',	'GATS2c',	'MATS4dv',	'AETA_eta_L',	'SM1_Dzpe',	'ATSC6s',	'BCUTv-1l',	'PEOE_VSA8',	'AETA_eta_B',	'ZMIC3',	'BCUTare-1h',	'AATSC2v',	'MATS1i',	'AATS3m',	'ATSC6c',	'VR3_A',	'BCUTZ-1h',	'Mse',	'AATSC5i',	'BCUTdv-1l',	'AATSC4v',	'MATS1m',	'ATSC1m',	'ATS3pe',	'ETA_dBeta',	'JGI1',	'BCUTpe-1h',	'NsNH2',	'AATSC0s',	'GATS3c',	'VR2_Dzi',	'ATSC2c',	'VSA_EState2',	'VE1_A',	'ATSC2pe',	'ATSC4i',	'Xc-4dv',	'ATSC0s',	'BIC4',	'ABC',	'ATS3se',	'ATSC4dv',	'GGI3',	'MATS1se',	'ATSC2Z',	'VAdjMat',	'ATSC3i',	'AATS0pe',	'PEOE_VSA5',	'AETA_beta',	'VR2_Dzm',	'nX',	'AATS4se',	'VE2_Dzse',	'ATSC0Z',	'SssNH',	'ATSC2v',	'MATS2pe',	'AETA_eta',	'ATSC5s',	'GATS3v',	'nAromAtom',	'IC4',	'PEOE_VSA4',	'AXp-3d',	'ATSC3se',	'AXp-1dv',	'MATS4p',	'AATS3s',	'ATSC7s',	'ATS2dv',	'ECIndex',	'GATS2are',	'AMID_C',	'AMID_h',	'SMR_VSA9',	'Kier1',	'SpMAD_Dzv',	'SMR_VSA1',	'ATS3m',	'Xp-3d',	'ATS5are',	'AATSC1d',	'EState_VSA9',	'MATS4i',	'AATSC5d',	'SM1_Dzare',	'MATS2se',	'ETA_epsilon_1',	'GATS3d',	'AATSC3d',	'GATS2v',	'ATSC3v',	'AATSC5dv',	'AATS1are',	'MATS5d',	'GATS1dv',	'AATS0i',	'ATS4dv',	'BCUTi-1h',	'GATS2Z',	'AETA_eta_RL',	'AATSC2are',	'AXp-0dv',	'SpDiam_A',	'JGI3',	'ATS3s',	'SaaCH',	'BCUTse-1h',	'GGI6',	'CIC3',	'ATS7se',	'piPC5',	'ATSC0are',	'AATSC2c',	'ATS2p',	'SdsN',	'MATS2c',	'AATS5Z',	'MATS1p',	'SpDiam_Dzp',	'IC3',	'AATS3dv',	'ATSC1dv',	'SpMax_A',	'SsssCH',	'AETA_beta_ns',	'AATS4i',	'Xpc-6d',	'ETA_eta_RL',	'GATS2se',	'GATS4p',	'ATS6se',	'VR1_A',	'MATS3pe',	'SpMAD_Dzi',	'SaaN',	'JGI9',	'GATS3i',	'ATSC2p',	'Radius',	'MATS1c',	'AATSC0i',	'Xp-2dv',	'Xc-3d',	'IC1',	'NaasC',	'AATSC5p',	'AMID',	'Xp-6dv',	'VR2_DzZ',	'AATSC0c',	'AATS3Z',	'RPCG',	'MATS4v',	'ATSC4p',	'AATS4d',	'ATS3Z',	'LogEE_Dzse',	'SpAbs_A',	'GATS2m',	'ETA_eta_BR',	'ATSC6i',	'ATSC5dv',	'SpAD_A',	'ATSC7i',	'MATS1s',	'JGT10',	'AATS2d',	'nBondsT',	'ATS1s',	'GATS1c',	'GATS4se',	'Xpc-4dv',	'piPC7',	'ATS8m',	'BIC5',	'SpMAD_D',	'ATSC5are',	'SpDiam_Dzv',	'AATSC4i',	'VSA_EState4',	'ATS0dv',	'MATS2dv',	'ATSC6pe',	'GATS3s',	'SpAbs_Dzp',	'AETA_beta_ns_d',	'AATSC2pe',	'MATS1dv',	'AETA_dBeta',	'n5HRing',	'AATS4pe',	'SIC1',	'naRing',	'AETA_eta_F',	'ATS6p',	'AATS1i',	'AATS3v',	'Mm',	'VMcGowan',	'SpAbs_Dzpe',	'ATS3i',	'ATSC6v',	'ATS5p',	'AATSC3p',	'ATS6m',	'SdsCH',	'apol',	'AATS5v',	'IC5',	'VE2_Dzpe',	'ETA_eta_L',	'SRW04',	'AXp-1d',	'SM1_Dzp',	'SlogP_VSA11',	'GATS4c',	'ATSC0p',	'MATS2d',	'ETA_eta',	'VR1_Dzse',	'EState_VSA5',	'VE1_Dzm',	'MATS5m',	'BCUTs-1h',	'Xp-1d',	'IC2',	'SpAD_Dzv',	'SpMax_Dzare',	'AATSC3dv',	'ATS8dv',	'PEOE_VSA9',	'BIC1',	'AATSC3Z',	'JGI4',	'AATSC3v',	'VR2_Dzp',	'AXp-5dv',	'AATSC1dv',	'VR3_Dzpe',	'n10FHRing',	'Mare',	'SpAD_Dzse',	'Kier2',	'SMR_VSA3',	'AATSC1p',	'PEOE_VSA10',	'AATS3i',	'Xp-7d',	'VE3_Dzm',	'ATSC6dv',	'MATS5Z',	'MID_X',	'ATSC2m',	'fragCpx',	'AATS1pe',	'ATS4pe',	'ATS4se',	'SIC2',	'naHRing',	'EState_VSA3',	'MATS3i',	'ATS2v',	'MATS3se',	'GATS3p',	'SMR',	'SpAD_Dzpe',	'VR1_DzZ',	'AATSC3pe',	'Mpe',	'AATSC2m',	'JGI10',	'StsC',	'ATS8v',	'SlogP_VSA6',	'Xch-7dv',	'AATS5se',	'VSA_EState7',	'SpMAD_A',	'ATSC8c',	'VR1_Dzare',	'ATS1v',	'ATS7i',	'VSA_EState3',	'ATSC4v',	'ATSC2s',	'ATSC5d',	'ATSC8dv',	'ATSC0se',	'ATSC8Z',	'AATS2se',	'GATS2p',	'GATS4pe',	'SpAD_D',	'ATS6i',	'ATS0p',	'GATS3se',	'AATS3p',	'AATS4dv',	'BCUTc-1l',	'ATS7p',	'ATSC1i',	'SsssN',	'VR3_Dzse',	'ATS6v',	'ATSC3dv',	'VSA_EState5',	'Zagreb1',	'AATS0s',	'VE2_D',	'LogEE_Dzi',	'AXp-0d',	'AATS2i',	'AATS0are',	'AATS3are',	'MID_O',	'AMID_N',	'AATS3pe',	'AATS2are',	'MID_N',	'AMID_O',	'AATS5are',	'AATS2p',	'nBondsO',	'AMID_X',	'ATS6s',	'GGI8',	'GGI9',	'GGI10',	'ATS2m',	'ATS1m',	'JGI2',	'JGI5',	'ATS5Z',	'ATS4Z',	'JGI6',	'ATS1Z',	'ATS7s',	'ATS0s',	'ATS7v',	'ATS7dv',	'ATS6dv',	'ATS3dv',	'ATS1dv',	'nHeavyAtom',	'MWC01',	'MWC02',	'VE3_A',	'VE2_A',	'SRW02',	'LogEE_A',	'SRW07',	'ATS0v',	'GGI7',	'AATS1se',	'n10FaRing',	'AATS0se',	'piPC1',	'AATS2v',	'AATS5m',	'piPC10',	'bpol',	'n6HRing',	'n6aRing',	'nAHRing.1',	'AATS5s',	'AATS4s',	'AATS1s',	'AATS3d',	'GGI4',	'AATS1d',	'AATS0d',	'AATS5dv',	'AATS2dv',	'AATS1dv',	'AATS0dv',	'ATS4i',	'ATS8p',	'TopoPSA',	'ATS7are',	'ATS3are',	'ATS7pe',	'ATS5i',	'CIC1',	'AATS5i',	'SpMax_Dzse',	'SM1_DzZ',	'Xp-7dv',	'Xp-4dv',	'SM1_Dzm',	'VE2_Dzm',	'VR1_Dzm',	'VR3_Dzm',	'SpAbs_Dzv',	'Xp-0dv',	'SM1_Dzv',	'AXp-2d',	'VE3_Dzv',	'VR1_Dzv',	'VR2_Dzv',	'VR3_Dzv',	'BalabanJ',	'BCUTp-1l',	'AXp-4dv',	'NtN',	'SdssC',	'GATS4v',	'GATS2pe',	'NsCl',	'GATS2i',	'NsssN',	'BCUTc-1h',	'Sv',	'BCUTdv-1h',	'NaaCH',	'VR1_D',	'BCUTs-1l',	'VE3_D',	'SpDiam_D',	'SpAbs_Dzse',	'SpDiam_Dzse',	'GATS4Z',	'SpMAD_Dzse',	'Xc-5dv',	'LogEE_Dzp',	'VE3_Dzp',	'Xch-7d',	'C3SP3',	'SpAbs_Dzi',	'SpMax_Dzi',	'SpDiam_Dzi',	'SpAD_Dzi',	'SM1_Dzi',	'VE1_Dzi',	'C1SP1',	'VE3_Dzi',	'VR1_Dzi',	'VR3_Dzi',	'SpMax_Dzp',	'VR3_Dzare',	'Xpc-5dv',	'VR1_Dzpe',	'SM1_Dzse',	'SpMax_Dzpe',	'SpDiam_Dzpe',	'SpMAD_Dzpe',	'LogEE_Dzpe',	'Xp-2d',	'VR2_Dzpe',	'VE2_Dzare',	'SpAbs_Dzare',	'SpDiam_Dzare',	'SpAD_Dzare',	'SpMAD_Dzare',	'LogEE_Dzare',	'Xpc-6dv',	'SsNH2',	'GATS4s',	'ATSC0c',	'CIC0',	'ATSC6are',	'ATSC8are',	'ATSC1p',	'PEOE_VSA1',	'ATSC5p',	'ATSC6p',	'ATSC2i',	'Kier3',	'ZMIC5',	'MIC1',	'AATSC2dv',	'MIC0',	'AATSC0d',	'AATSC2d',	'CIC5',	'ATSC3are',	'ATSC8pe',	'PEOE_VSA7',	'ATSC8d',	'ATSC5c',	'MID',	'ATSC2dv',	'ATSC0d',	'ATSC2d',	'ATSC7d',	'ATSC4s',	'ATSC3pe',	'EState_VSA10',	'EState_VSA8',	'ATSC4Z',	'SlogP_VSA8',	'SlogP_VSA1',	'ATSC8se',	'AATSC3s',	'AATSC1Z',	'SddsN',	'BIC2',	'MATS2Z',	'ETA_eta_F',	'MATS2m',	'AETA_eta_R',	'ETA_eta_R',	'MATS2v',	'MATS3v',	'MATS2are',	'MATS2p',	'MATS3p',	'ETA_beta_s',	'GATS2dv',	'ETA_shape_y',	'GATS5d',	'SdO',	'ETA_dAlpha_B',	'MATS3d',	'MATS5dv',	'IC0',	'SIC4',	'TIC3',	'AATSC5v',	'AATSC1se',	'AATSC2se',	'AATSC0are',	'AATSC3are',	'ETA_epsilon_4',	'AATSC2p',	'AATSC4p',	'AATSC1i',	'ETA_dPsi_A',	'AATSC3i',	'ETA_dEpsilon_A',	'mZagreb2']
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Check if all required columns are present
    if not all(col in df.columns for col in columns):
        raise ValueError("Not all specified columns exist in the CSV file.")
    data = df[columns]
    
    # Load the StandardScaler from pickle file
    scaler = joblib.load(scaler_pkl)
    
    data_normalized = pd.DataFrame(scaler.transform(data), columns=data.columns)
    # data_normalized.to_csv("normalized_data.csv", index=False)
    
    # Load the predictive model from another pickle file
    model = joblib.load(model_pkl)
    
    # Make predictions
    predictions = model.predict(data_normalized)
    probabilities = model.predict_proba(data_normalized)[:, 1]  # Probability of being 1
    labels = ['Permeable' if x == 1 else 'Impermeable' for x in predictions]
    
    # Create a DataFrame for output
    result_df = pd.DataFrame({
        'Predictions': labels,
        'Probability_of_Permeability': probabilities
    })
    result_df.to_csv("Prediction_Results.csv", index=False)
    return result_df

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <path_to_csv_file> <path_to_scaler_pkl> <path_to_model_pkl>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    scaler_pkl = sys.argv[2]
    model_pkl = sys.argv[3]
    
    try:
        output_df = preprocess_and_predict(csv_path, scaler_pkl, model_pkl)
        print(output_df)
    except Exception as e:
        print(f"An error occurred: {e}")
