import numpy as np
import pandas as pd


data = pd.DataFrame({"Sốt": ["Cao", "Cao", "Cao", "Thấp", "Cao", "Thấp", "Thấp", "Cao", "Th ấp", "Cao"],
"Ho": ["Có", "Không", "Có", "Có", "Có", "Không", "Có", "Không", "Không", "Có"],
"Đau họng": ["Có", "Có", "Không", "Có", "Có", "Có", "Không", "Có", "Kh ông", "Có"],
"Mệt mỏi": ["Có", "Không", "Có", "Có", "Không", "Không", "Có", "Có", " Không", "Có"],
"Flu": ["Có", "Có", "Có", "Không", "Có", "Không", "Không", "Có", "Khô ng", "Có"]
})


new_patient = {"Sốt": "Cao", "Ho": "Có", "Đau họng": "Có", "Mệt mỏi": "Không"}

P_flu = data["Flu"].value_counts(normalize=True)["Có"] 
P_not_flu = 1- P_flu 

def conditional_prob(feature, value, flu_status):
    data_filtered = data[data["Flu"] == flu_status]
    return data[feature].value_counts(normalize=True)[value] 


# Tính P(X|C) = P(x_1|C) * P(x_2|C) * ... * P(x_n|C) theo Naive Bayes
P_X_given_flu = 1
P_X_given_not_flu = 1 

for feature, value in new_patient.items():
    P_X_given_flu *= conditional_prob(feature, value, "Có")
    P_X_given_not_flu *= conditional_prob(feature, value, "Không")

# Tính P(C|X) trực tiếp từ tỷ lệ (không cần P(X))
numerator_flu = P_X_given_flu * P_flu
numerator_not_flu = P_X_given_not_flu * P_not_flu

total = numerator_flu + numerator_not_flu
P_flu_given_X = numerator_flu / total
P_not_flu_given_X = numerator_not_flu / total
print("\nKết quả Naive Bayes:")
print(f"P(Flu=Có|X) = {P_flu_given_X:.4f}")
print(f"P(Flu=Không|X) = {P_not_flu_given_X:.4f}")
print("\nKết luận:", "Bệnh nhân có khả năng bị cúm." if P_flu_given_X >
          
P_not_flu_given_X else "Bệnh nhân KHÔNG có khả năng bị cúm.")