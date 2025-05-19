import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, hamming_loss, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn import tree
from sklearn.tree import plot_tree, DecisionTreeClassifier, DecisionTreeRegressor

df = pd.read_csv("plasticwaste.csv")

df[df['Recycling_Rate'] == df['Recycling_Rate'].max()] #País com o maior índice de reciclagem
df[df['Recycling_Rate'] == df['Recycling_Rate'].min()] #País com o menor índice de reciclagem
df['Coastal_Waste_Risk'].value_counts() #Classificações de risco costal nos países
df[df['Coastal_Waste_Risk'] == "Very_High"] #Quais países possuem risco costal muito alto?
df['Main_Sources'].value_counts() #Quais as principais fontes de resíduos de plástico nos países?

df['Coastal_Risk_Label'] = LabelEncoder().fit_transform(df['Coastal_Waste_Risk'])
x = df[['Recycling_Rate','Per_Capita_Waste_KG','Total_Plastic_Waste_MT']]
y = df['Coastal_Risk_Label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42, max_depth = 3)
dt_model.fit(x_train, y_train)
dt_pred = dt_model.predict(x_test)
plt.figure(figsize=(60,45))
tree.plot_tree(dt_model,
          filled=True,
          rounded=True,
          feature_names=['Recycling_Rate','Per_Capita_Waste_KG','Total_Plastic_Waste_MT'],
          class_names=encoder.classes_)
report_tree = classification_report(y_test, dt_pred)
print(report_tree)

#Agrupa os dados por país e soma a quantidade total de plástico
Qnt_Total_Pais = df.groupby("Country")["Total_Plastic_Waste_MT"].sum()

# Seleciona os 15 países com maior produção
top10 = Qnt_Total_Pais.sort_values(ascending=False).head(15)

# Criando o gráfico de barras:
plt.figure(figsize=(15, 6))
top10.plot(kind="bar", color="skyblue")
plt.title("Top 15 Países que mais produzem resíduos plásticos")
plt.xlabel("País")
plt.ylabel("Resíduos plásticos (milhões de toneladas)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Conta as ocorrências de cada "Main_Sources" no conjunto de dados
source_counts = df['Main_Sources'].value_counts()

# Mostra o resultado
mais_comum = source_counts.idxmax()  # Fonte mais comum
quantidade = source_counts.max()  # Número de vezes que ela aparece

print(f"A fonte de residuos mais comum é '{mais_comum}' e aparece {quantidade} vezes.")

# Conta as ocorrências de cada "Main_Sources" no conjunto de dados
source_counts = df['Main_Sources'].value_counts()

# Mostra o resultado
mais_comum = source_counts.idxmax()  # Fonte mais comum
quantidade = source_counts.max()  # Número de vezes que ela aparece

print(f"A fonte de residuos mais comum é '{mais_comum}' e aparece {quantidade} vezes.")

# Ordena os países pela taxa de reciclagem (decrescente para maiores taxas e crescente para menores)
maiores_taxas_reciclagem = df.sort_values(by='Recycling_Rate', ascending=False)
menores_taxas_reciclagem = df.sort_values(by='Recycling_Rate', ascending=True)

# Exibe os 5 países com maiores e menores taxas de reciclagem
print("Maiores taxas de reciclagem:")
print(maiores_taxas_reciclagem[['Country', 'Recycling_Rate', 'Per_Capita_Waste_KG']].head())

print("\nMenores taxas de reciclagem:")
print(menores_taxas_reciclagem[['Country', 'Recycling_Rate', 'Per_Capita_Waste_KG']].head())

# Analisa a correlação entre a taxa de reciclagem e resíduos per capita
correlacao = df[['Recycling_Rate', 'Per_Capita_Waste_KG']].corr()

print("\nCorrelação entre a taxa de reciclagem e os resíduos per capita:")
print(correlacao)

# Selecionar os 30 países com maior produção total de resíduos plásticos
top30 = df.sort_values(by="Total_Plastic_Waste_MT", ascending=False).head(30)

# Paleta de cores para risco costeiro
risk_palette = {
    "Low": "green",
    "Medium": "orange",
    "High": "red",
    "Very_High": "darkred"
}

# Criar o gráfico
plt.figure(figsize=(14, 8))
scatter = sns.scatterplot(
    data=top30,
    x="Per_Capita_Waste_KG",
    y="Total_Plastic_Waste_MT",
    hue="Coastal_Waste_Risk",
    size="Recycling_Rate",
    sizes=(40, 400),
    palette=risk_palette,
    alpha=0.7,
    edgecolor="black",
    legend="full"
)

# Adicionar os nomes dos países
for i in range(len(top30)):
    plt.text(
        top30["Per_Capita_Waste_KG"].iloc[i],
        top30["Total_Plastic_Waste_MT"].iloc[i],
        top30["Country"].iloc[i],
        fontsize=8,
        ha='right'
    )

# Títulos e ajustes
plt.title("Top 30 Países por Resíduo Plástico Total: Relação com Reciclagem e Risco Costeiro", fontsize=16)
plt.xlabel("Resíduo Per Capita (kg)")
plt.ylabel("Resíduo Total (milhões de toneladas)")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()

# Mostrar o gráfico
plt.show()
# Encontrar a maior taxa de reciclagem
max_recycling_rate = df["Recycling_Rate"].max()

# Calcular reciclado atualmente e simulado
df["Current_Recycled_MT"] = df["Total_Plastic_Waste_MT"] * (df["Recycling_Rate"] / 100)
df["Simulated_Recycled_MT"] = df["Total_Plastic_Waste_MT"] * (max_recycling_rate / 100)

# Totais globais
total_current_recycled = df["Current_Recycled_MT"].sum()
total_simulated_recycled = df["Simulated_Recycled_MT"].sum()
recycling_gain = total_simulated_recycled - total_current_recycled
percent_gain = (recycling_gain / total_current_recycled) * 100

# Dados para o gráfico
categories = ["Reciclado Atual", "Reciclado com Alta Eficiência"]
values = [total_current_recycled, total_simulated_recycled]

# Criar gráfico de barras
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=["#1f77b4", "#2ca02c"])
plt.title("Impacto Global de Reciclagem com Alta Eficiência", fontsize=14)
plt.ylabel("Milhões de Toneladas Recicladas")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionar valores numéricos nas barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 5,
             f'{height:.1f}', ha='center', va='bottom', fontsize=12)

# Adicionar legenda extra com economia percentual
plt.text(0.5, max(values) + 30,
         f"Ganho potencial: +{recycling_gain:.1f} Mt ({percent_gain:.1f}% a mais)",
         ha='center', fontsize=12, fontweight='bold', color='darkgreen')

plt.tight_layout()
plt.show()
