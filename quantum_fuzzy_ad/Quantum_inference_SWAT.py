# import libraries
import pandas as pd
import numpy as np
import re


import matplotlib.pyplot as plt
import matplotlib

import seaborn as sns
from pylab import rcParams
from sklearn.metrics import mean_squared_error, roc_curve, auc
from qiskit_ibm_runtime import QiskitRuntimeService
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz
import QFIE.FuzzyEngines as FE
from skfuzzy.defuzzify.exceptions import EmptyMembershipError
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8


import membership
import rule_base
import mamdani
import wangmendel
import defuzz
from Examples.MackeyGlass import synthetic

if __name__ == "__main__":

    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token="b7afa0f6888a2e6bff61cf4fd18c145d7f8cf5edf7981f8a2878e166221e621fd66aa7e40c9c5c89a9f41265f5f304b96c17ae5d44b98802f976a31f0c311b56",
        set_as_default=True,
        # Use `overwrite=True` if you're updating your token.
        overwrite=True,
    )

    service = QiskitRuntimeService()
    backend = service.backend("ibm_sherbrooke")

        # 🔹 Carica il dataset (se non è già in memoria)
    df = pd.read_csv("./data/relevant_features.csv")
    df = df.iloc[:, [0, 1, -1]] 

    # 🔹 Seleziona solo le feature di input (senza OUTCOME)
    X = df.drop(columns=["OUTCOME"])
    y = df["OUTCOME"]

    # 🔹 Filtra solo dati normali per training e validation set
    df_normal = df[df["OUTCOME"] == 0]
    df_anomaly = df[df["OUTCOME"] == 1]

    # 🔹 Prendi i primi 9000 dati normali per il training set
    X_train = df_normal.iloc[:9000, :-1].values  # Escludi OUTCOME
    y_train = df_normal.iloc[:9000, -1].values  # Target

    # 🔹 Prendi i successivi 9000 dati normali per il validation set
    X_val = df_normal.iloc[9000:18000, :-1].values
    y_val = df_normal.iloc[9000:18000, -1].values

    X_train = np.vstack((X_train, X_val))
    y_train = np.hstack((y_train, y_val))

    X_test_5perc = pd.concat([
        df_normal.iloc[19750:23075].drop(columns=["OUTCOME"]), 
        df_anomaly.iloc[1750:1925].drop(columns=["OUTCOME"])
    ]).values

    y_test_5perc = pd.concat([
        df_normal.iloc[19750:23075]["OUTCOME"], 
        df_anomaly.iloc[1750:1925]["OUTCOME"]
    ]).values

    # 🔹 Normalizza con MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(df)  # Ora il numero di feature è coerente
    X_train = scaler.fit_transform(X_train)
    X_test_5perc = pd.DataFrame(X_test_5perc)
    X_test_5perc = scaler.transform(X_test_5perc)

    A = np.array(X_train)[:,0]
    print(A.shape)

    n_regions = 3
    name_preffix = 'AIT201'

    rule_base = wangmendel.learn_fuzzy_rules(X_train, y_train,
                                        n_regions_inputs=[3,3],
                                        n_regions_output=2,
                                        name_preffix_inputs=['AIT201', 'AIT501'],
                                        name_preffix_output='Y')

    print("The rule base has {} rules!".format(rule_base.size()))

    rule_base = wangmendel.clean_rule_base(rule_base)
    print("The cleaned rule base has {} rules!".format(rule_base.size()))

    rule_list = []
    
    for i in range(rule_base.size()):
        rule_list.append(rule_base.printRule(i))
    
    def map_fuzzy_set_antecedent(fuzzy_set):
            if "triang" in fuzzy_set.__str__():
                return "medium"
            elif "inf_border" in fuzzy_set.__str__():
                return "low"
            elif "sup_border" in fuzzy_set.__str__():
                return "high"

    linguistic_rules = []

    for antecedent_list, consequent_data, strength in rule_list:
        antecedent_str = " and ".join([f"{var} is {map_fuzzy_set_antecedent(fuzzy_set)}" for var, fuzzy_set in antecedent_list])
        consequent_var, consequent_fuzzy_set = consequent_data
        consequent_str = f"{consequent_var} is normal"
        
        rule_str = f"if {antecedent_str} then {consequent_str}"
        linguistic_rules.append(rule_str)


    def remove_last_digit_from_antecedent(rule):
        # Separiamo l'antecedente e il conseguente
        antecedent, consequent = rule.split(' then ')
        # Rimuoviamo l'ultimo numero dai sensori solo nell'antecedente
        antecedent = re.sub(r'(\b[A-Za-z]+\d+)(?=\s)', lambda m: m.group(0)[:-1], antecedent)
        # Ricostruiamo la regola
        return antecedent + ' then ' + consequent

    # Applica la funzione a tutte le regole
    rules = [remove_last_digit_from_antecedent(rule) for rule in linguistic_rules]
    rules = list(set([rule.strip() for rule in rules]))

    sens_1 = np.linspace(-0.00567416, 0.54546422, 10)
    sens_2 = np.linspace(0.04376599, 0.1238632, 10)
    Y1 = np.linspace(0, 1, 1)


    # Aggiunto un margine del 5% ai valori minimi e massimi
    margin_sens1 = 0.05 * (0.54546422 - (-0.00567416))
    margin_sens2 = 0.05 * (0.1238632 - 0.04376599)

    # Intervalli rilassati per sens_1
    a_sens1 = -0.00567416 - margin_sens1
    b_sens1 = 0.3  
    c_sens1 = 0.54546422 + margin_sens1  

    # Intervalli rilassati per sens_2
    a_sens2 = 0.04376599 - margin_sens2
    b_sens2 = 0.085  
    c_sens2 = 0.1238632 + margin_sens2  

    # Funzioni di appartenenza per sens_1 (con intervalli rilassati)
    mf1_sens1 = fuzz.trimf(sens_1, [a_sens1, a_sens1, b_sens1])  
    mf2_sens1 = fuzz.trimf(sens_1, [b_sens1, (b_sens1 + c_sens1) / 2, c_sens1])  
    mf3_sens1 = fuzz.trimf(sens_1, [0.8 * c_sens1, c_sens1, c_sens1 + margin_sens1])  

    # Funzioni di appartenenza per sens_2 (con intervalli rilassati)
    mf1_sens2 = fuzz.trimf(sens_2, [a_sens2, a_sens2, b_sens2])  
    mf2_sens2 = fuzz.trimf(sens_2, [b_sens2, (b_sens2 + c_sens2) / 2, c_sens2])  
    mf3_sens2 = fuzz.trimf(sens_2, [0.9 * c_sens2, c_sens2, c_sens2 + margin_sens2]) 

    mf1_Y1 = fuzz.trimf(Y1, [0, 0, 1])  # [basso, basso, medio] - Normale


    qfie = FE.QuantumFuzzyEngine(verbose=False)
    qfie.input_variable(name='AIT201', range=sens_1)
    qfie.input_variable(name='AIT501', range=sens_2)
    qfie.output_variable(name='Y1', range=Y1)
    qfie.add_input_fuzzysets(var_name='AIT201', set_names=['low', 'medium', 'high'], sets=[mf1_sens1, mf2_sens1, mf3_sens1])
    qfie.add_input_fuzzysets(var_name='AIT501', set_names=['low', 'medium', 'high'], sets=[mf1_sens2, mf2_sens2, mf3_sens2])
    qfie.add_output_fuzzysets(var_name='Y1', set_names=['normal'],sets=[mf1_Y1])

    #RULES GENERATED BY PSO (TRAINING) and PATTERN SEARCH (TUNING):
    rules_1 = ['if AIT201 is low and AIT501 is high then Y1 is normal',
               'if AIT501 is high then Y1 is normal',
               'if AIT201 is high then Y1 is normal',
               'if AIT201 is low then Y1 is normal'
               ]
    #RULES GENERATED BY PSO (TRAINING) and Simulated Annealing (TUNING):

    rules_2 = ['if AIT201 is low and AIT501 is low then Y1 is normal',
               'if AIT201 is low then Y1 is normal',
               'if AIT201 is low and AIT501 is medium then Y1 is normal']
    
    #RULES GENERATED BY Genetic Algorithm (TRAINING) and Pattern Search (TUNING):
    
    rules_3 =  ['if AIT201 is low and AIT501 is high then Y1 is normal',
               'if AIT501 is high then Y1 is normal',
               'if AIT201 is high then Y1 is normal',
               'if AIT201 is low then Y1 is normal']

    #RULES GENERATED BY Genetic Algorithm (TRAINING) and Simulated Annealing(TUNING):

    rules_4 = ['if AIT201 is low and AIT501 is low then Y1 is normal',
               'if AIT201 is low then Y1 is normal',
               'if AIT201 is low and AIT501 is medium then Y1 is normal',]

    qfie.set_rules(rules_4)

    f_quantum = []
    i = 0
    errors = []
    # Threshold FISSO per rilevare anomalie basato sull'errore di ricostruzione
    THRESHOLD = 0.05 
    
    for row in X_test_5perc: 
        
        input_values = {
            'AIT201': row[0],
            'AIT501': row[1]
        }
        
        
        # Costruct Quantum Circuit for Inference
        qfie.build_inference_qc(input_values, draw_qc=True, filename=f"q_rules/quantum_SWAT_rule_rc_4.pdf")
        i = i + 1
        
        try:
            # Quantum Inference
            predicted_value = qfie.execute(n_shots=1000, plot_histo=True, backend=backend)[0]

            # Calcolo dell'errore di ricostruzione rispetto all'etichetta reale
            error = mean_squared_error([y_test_5perc[i - 1]], [predicted_value])

            # Salvataggio dell'errore
            errors.append(error)

            # Se l'errore supera la soglia fissa, assegniamo l'etichetta 1 (anomalia)
            if error > THRESHOLD:
                result = 1
            else:
                result = predicted_value
        except EmptyMembershipError:
            # Se l'errore EmptyMembershipError si verifica, assegniamo direttamente l'etichetta 1
            result = 1
            errors.append(THRESHOLD + 0.01)  # Se l'errore si verifica, inseriamo l'etichetta 1

        f_quantum.append(result)
    
    fpr, tpr, _ = roc_curve(y_test_5perc, np.array(f_quantum))
    roc_auc = auc(fpr, tpr)

    # Disegno della curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Linea diagonale per riferimento
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("q_rules/roc_curve_rc_4.png", dpi=300, bbox_inches='tight')


    
