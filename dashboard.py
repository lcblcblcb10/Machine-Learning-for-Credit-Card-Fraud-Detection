import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Supressão de warnings
import warnings
warnings.filterwarnings('ignore')

# Definição das cores
colors = ["#0101DF", "#DF0101"]

# Função para carregar os dados
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

# Carregando os dados
st.title("Análise de Transações com Cartões de Crédito")
st.sidebar.header("Configurações")
file_path = st.sidebar.text_input("Insira o caminho do arquivo CSV:", r'C:\Users\AGFAKZZ\Desktop\FIAP\Fase 3 - Arquitetura ML e Aprendizado\creditcard.csv\creditcard.csv')

if file_path:
    try:
        df = load_data(file_path)
        st.success("Dados carregados com sucesso!")
        
        # Exibindo informações sobre os dados
        st.subheader("Visualização inicial dos dados")
        st.write(df.head())
        
        # Criando o gráfico
        st.subheader("Distribuição das Classes (Fraude vs Não Fraude)")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.countplot(x='Class', data=df, palette=colors, ax=ax)
        ax.set_title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
else:
    st.warning("Por favor, insira o caminho do arquivo CSV para começar.")

# Criando o gráfico de distribuições
st.subheader("Distribuições de Valores e Tempo das Transações")
fig, ax = plt.subplots(1, 2, figsize=(18, 4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.histplot(amount_val, ax=ax[0], color='r', kde=True)
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.histplot(time_val, ax=ax[1], color='b', kde=True)
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

st.pyplot(fig)

# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)
from sklearn.preprocessing import StandardScaler, RobustScaler

# RobustScaler is less prone to outliers.

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

# Amount and Time are Scaled!

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.
# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the Distribution of the labels


# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

# Since our classes are highly skewed we should make them equivalent in order to have a normal distribution of the classes.

# Lets shuffle the data before creating the subsamples

df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

import seaborn as sns
import matplotlib.pyplot as plt

# Distribuição das classes no novo dataset balanceado
st.subheader("Distribuição das Classes no Subconjunto Balanceado")
fig, ax = plt.subplots(figsize=(10, 5))

sns.countplot(x='Class', data=new_df, palette=colors, ax=ax)
ax.set_title('Equally Distributed Classes', fontsize=14)

st.pyplot(fig)

# Boxplots das variáveis com correlação negativa com a classe
st.subheader("Boxplots das Variáveis com Correlação Negativa com a Classe")
fig, axes = plt.subplots(ncols=4, figsize=(20, 4))

sns.boxplot(x="Class", y="V17", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

st.pyplot(fig)

# Boxplots das variáveis com correlação positiva com a classe
st.subheader("Boxplots das Variáveis com Correlação Positiva com a Classe")
fig, axes = plt.subplots(ncols=4, figsize=(20, 4))

sns.boxplot(x="Class", y="V11", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V4 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V2", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V2 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V19", data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V19 vs Class Positive Correlation')

st.pyplot(fig)

# Distribuição das variáveis para transações fraudulentas
st.subheader("Distribuição das Variáveis para Transações Fraudulentas")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class'] == 1].values
sns.histplot(v14_fraud_dist, ax=axes[0], kde=True, stat="density", color='#FB8861', line_kws={"linewidth": 2})
axes[0].set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = new_df['V12'].loc[new_df['Class'] == 1].values
sns.histplot(v12_fraud_dist, ax=axes[1], kde=True, stat="density", color='#56F9BB', line_kws={"linewidth": 2})
axes[1].set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)

v10_fraud_dist = new_df['V10'].loc[new_df['Class'] == 1].values
sns.histplot(v10_fraud_dist, ax=axes[2], kde=True, stat="density", color='#C5B3F9', line_kws={"linewidth": 2})
axes[2].set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

st.pyplot(fig)

# Boxplots com remoção de outliers
st.subheader("Boxplots com Redução de Outliers")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

colors = ['#B3F9C5', '#f9c5b3']

# Feature V14
sns.boxplot(x="Class", y="V14", data=new_df, ax=axes[0], palette=colors)
axes[0].set_title("V14 Feature \n Reduction of outliers", fontsize=14)
axes[0].annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),
                 arrowprops=dict(facecolor='black'),
                 fontsize=12)

# Feature V12
sns.boxplot(x="Class", y="V12", data=new_df, ax=axes[1], palette=colors)
axes[1].set_title("V12 Feature \n Reduction of outliers", fontsize=14)
axes[1].annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),
                 arrowprops=dict(facecolor='black'),
                 fontsize=12)

# Feature V10
sns.boxplot(x="Class", y="V10", data=new_df, ax=axes[2], palette=colors)
axes[2].set_title("V10 Feature \n Reduction of outliers", fontsize=14)
axes[2].annotate('Fewer extreme \n outliers', xy=(0.95, -16.5), xytext=(0, -12),
                 arrowprops=dict(facecolor='black'),
                 fontsize=12)

st.pyplot(fig)

import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

# New_df is from the random undersample data (fewer instances)
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# T-SNE Implementation
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()

# PCA Implementation
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
t1 = time.time()

# Clusters usando Redução de Dimensionalidade
st.subheader("Clusters usando Redução de Dimensionalidade")
import matplotlib.patches as mpatches

# Configuração das legendas
blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

fig, axes = plt.subplots(1, 3, figsize=(24, 6))
fig.suptitle('Clusters using Dimensionality Reduction', fontsize=14)

# t-SNE scatter plot
axes[0].scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=y, cmap='coolwarm', linewidths=2)
axes[0].set_title('t-SNE', fontsize=14)
axes[0].grid(True)
axes[0].legend(handles=[blue_patch, red_patch])

# PCA scatter plot
axes[1].scatter(X_reduced_pca[:, 0], X_reduced_pca[:, 1], c=y, cmap='coolwarm', linewidths=2)
axes[1].set_title('PCA', fontsize=14)
axes[1].grid(True)
axes[1].legend(handles=[blue_patch, red_patch])

# Truncated SVD scatter plot
axes[2].scatter(X_reduced_svd[:, 0], X_reduced_svd[:, 1], c=y, cmap='coolwarm', linewidths=2)
axes[2].set_title('Truncated SVD', fontsize=14)
axes[2].grid(True)
axes[2].legend(handles=[blue_patch, red_patch])

st.pyplot(fig)

# Undersampling before cross validating (prone to overfit)
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# Our data is already scaled we should split our training and test sets
from sklearn.model_selection import train_test_split

# This is explicitly used for undersampling.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Turn the values into an array for feeding the classification algorithms.
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Let's implement simple classifiers

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

# Use GridSearchCV to find the best parameters.
from sklearn.model_selection import GridSearchCV


# Logistic Regression 
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
# We automatically get the logistic regression with the best parameters.
log_reg = grid_log_reg.best_estimator_

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
# KNears best estimator
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier
svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)

# SVC best estimator
svc = grid_svc.best_estimator_

# DecisionTree Classifier
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)

# tree best estimator
tree_clf = grid_tree.best_estimator_

from imblearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import NearMiss

# Define StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# We will undersample during cross-validation
undersample_X = df.drop('Class', axis=1)
undersample_y = df['Class']

# For splitting data during cross-validation
for train_index, test_index in sss.split(undersample_X, undersample_y):
    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]

# Convert data to arrays for consistency
undersample_Xtrain = undersample_Xtrain.values
undersample_Xtest = undersample_Xtest.values
undersample_ytrain = undersample_ytrain.values
undersample_ytest = undersample_ytest.values

# Lists to store metrics
undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []

# Implementing NearMiss Technique 
# Distribution of NearMiss (Just to see how it distributes the labels we won't use these variables)
X_nearmiss, y_nearmiss = NearMiss().fit_resample(undersample_X.values, undersample_y.values)

# Cross Validating the right way
log_reg = LogisticRegression()  # Example classifier (can replace with other classifiers)
for train, test in sss.split(undersample_Xtrain, undersample_ytrain):
    # Create pipeline using NearMiss and Logistic Regression
    undersample_pipeline = make_pipeline(NearMiss(sampling_strategy='majority'), log_reg)
    
    # Fit model on the training data
    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])
    
    # Make predictions on the test data
    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])
    
    # Append metrics to lists
    undersample_accuracy.append(undersample_pipeline.score(undersample_Xtrain[test], undersample_ytrain[test]))
    undersample_precision.append(precision_score(undersample_ytrain[test], undersample_prediction))
    undersample_recall.append(recall_score(undersample_ytrain[test], undersample_prediction))
    undersample_f1.append(f1_score(undersample_ytrain[test], undersample_prediction))
    undersample_auc.append(roc_auc_score(undersample_ytrain[test], undersample_prediction))

# Let's Plot LogisticRegression Learning Curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

def plot_learning_curve_streamlit(estimator1, estimator2, estimator3, estimator4, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14), sharey=True)
    if ylim is not None:
        plt.ylim(*ylim)
    
    # First Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")

    # Second Estimator 
    train_sizes, train_scores, test_scores = learning_curve(
        estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax2.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax2.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax2.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax2.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax2.set_title("KNeighbors Classifier Learning Curve", fontsize=14)
    ax2.set_xlabel('Training size (m)')
    ax2.set_ylabel('Score')
    ax2.grid(True)

    # Third Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax3.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax3.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax3.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax3.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax3.set_title("SVC Learning Curve", fontsize=14)
    ax3.set_xlabel('Training size (m)')
    ax3.set_ylabel('Score')
    ax3.grid(True)
    ax3.legend(loc="best")

    # Fourth Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax4.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax4.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax4.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax4.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax4.set_title("Decision Tree Classifier Learning Curve", fontsize=14)
    ax4.set_xlabel('Training size (m)')
    ax4.set_ylabel('Score')
    ax4.grid(True)
    ax4.legend(loc="best")
    return f

# Configuração do Streamlit
st.title("Learning Curves para Diferentes Modelos")
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

# Gerar os gráficos no Streamlit
fig = plot_learning_curve_streamlit(log_reg, knears_neighbors, svc, tree_clf, X_train, y_train, (0.87, 1.01), cv=cv, n_jobs=4)
st.pyplot(fig)

from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
# Create a DataFrame with all the scores and the classifiers names.

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,
                             method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

svc_pred = cross_val_predict(svc, X_train, y_train, cv=5,
                             method="decision_function")

tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)

log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Adicionando título na aplicação
st.title("Comparação de Curvas ROC para Classificadores")

# Função para gerar o gráfico da curva ROC
def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(16, 8))
    plt.title('ROC Curve \n Top 4 Classifiers', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                 arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                 )
    plt.legend()

    # Renderizar no Streamlit
    st.pyplot(plt)

# Chamando a função para plotar as curvas
graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr)

# Adicionando título na aplicação
st.title("Curva ROC - Regressão Logística")

# Função para plotar a curva ROC da Regressão Logística
def logistic_roc_curve(log_fpr, log_tpr):
    plt.figure(figsize=(12, 8))
    plt.title('Logistic Regression ROC Curve', fontsize=16)
    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2, label="Logistic Regression")
    plt.plot([0, 1], [0, 1], 'r--', label="Random Guess")
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.axis([-0.01, 1, 0, 1])
    plt.legend()

# Chamando a função para criar o gráfico
logistic_roc_curve(log_fpr, log_tpr)

# Exibindo o gráfico no Streamlit
st.pyplot(plt)

from sklearn.metrics import precision_recall_curve
precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
y_pred = log_reg.predict(X_train)

undersample_y_score = log_reg.decision_function(original_Xtest)

from sklearn.metrics import average_precision_score
undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,3))

precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)

import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Adicionando título na aplicação
st.title("Curva Precision-Recall - UnderSampling")

# Função para plotar a curva Precision-Recall
def plot_precision_recall_curve(original_ytest, undersample_y_score):
    precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)
    undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)
    
    # Criar a figura
    plt.figure(figsize=(12, 3))
    plt.step(recall, precision, color='#004a93', alpha=0.8, where='post', label="Precision-Recall Curve")
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='#48a6ff')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'UnderSampling Precision-Recall curve:\n Average Precision-Recall Score ={0:0.2f}'.format(
            undersample_average_precision
        ),
        fontsize=16
    )
    plt.legend()

# Plotando o gráfico
plot_precision_recall_curve(original_ytest, undersample_y_score)

# Exibindo o gráfico no Streamlit
st.pyplot(plt)

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, Dropout, Flatten  # or whatever layers you need
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

n_inputs = X_train.shape[1]

undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

from tensorflow.keras.optimizers import Adam

# Corrected optimizer instantiation
undersample_model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=2)

undersample_predictions = undersample_model.predict(original_Xtest, batch_size=200, verbose=0)

# Predict probabilities for each class
predictions = undersample_model.predict(original_Xtest, batch_size=200, verbose=0)

# Convert probabilities to class labels (0 or 1, for binary classification)
undersample_fraud_predictions = (predictions > 0.5).astype(int)

# If you have more than 2 classes, you would use np.argmax instead to get the index of the highest probability class.

import itertools

# Adicionando título no aplicativo Streamlit
st.title("Matrizes de Confusão")

# Função para plotar as matrizes de confusão
def plot_confusion_matrices():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Subplot para Random UnderSample Confusion Matrix
    sns.heatmap(undersample_cm, annot=True, fmt="d", cmap=plt.cm.Reds, xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Random UnderSample \n Confusion Matrix", fontsize=16)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # Subplot para Confusion Matrix (100% Accuracy)
    sns.heatmap(actual_cm, annot=True, fmt="d", cmap=plt.cm.Greens, xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Confusion Matrix \n (with 100% Accuracy)", fontsize=16)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    return fig

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Convert original_ytest to binary format if it's one-hot encoded
if original_ytest.ndim > 1 and original_ytest.shape[1] > 1:
    original_ytest = np.argmax(original_ytest, axis=1)

# If undersample_fraud_predictions contains probabilities, convert to binary labels
if undersample_fraud_predictions.ndim > 1:
    undersample_fraud_predictions = np.argmax(undersample_fraud_predictions, axis=1)

# Generate confusion matrices
undersample_cm = confusion_matrix(original_ytest, undersample_fraud_predictions)
actual_cm = confusion_matrix(original_ytest, original_ytest)

labels = ['No Fraud', 'Fraud']

# Gerar as figuras e exibir no Streamlit
fig = plot_confusion_matrices()
st.pyplot(fig)