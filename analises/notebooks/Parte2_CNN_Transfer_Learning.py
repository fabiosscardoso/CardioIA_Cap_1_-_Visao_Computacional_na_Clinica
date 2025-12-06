"""
PARTE 2: Classificação de Imagens Médicas com CNN
CardioIA - A Nova Era da Cardiologia Inteligente

Este notebook implementa dois modelos de classificação:
1. CNN Simples (treinado do zero)
2. Transfer Learning com VGG16 e ResNet50

Inclui:
- Definição e treinamento dos modelos
- Avaliação com métricas (acurácia, matriz de confusão, precisão, recall, F1-score)
- Visualização dos resultados
- Comparação entre os modelos
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

# TensorFlow e Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)

# Configurações
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 80)
print("PARTE 2: CLASSIFICAÇÃO DE IMAGENS MÉDICAS COM CNN")
print("CardioIA - A Nova Era da Cardiologia Inteligente")
print("=" * 80)
print(f"\nData de execução: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print(f"Versão TensorFlow: {tf.__version__}")
print(f"GPU Disponível: {tf.config.list_physical_devices('GPU')}\n")

# ============================================================================
# SEÇÃO 1: CARREGAMENTO DOS DADOS PRÉ-PROCESSADOS
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 1: CARREGAMENTO DOS DADOS PRÉ-PROCESSADOS")
print("=" * 80)

data_dir = Path('/home/ubuntu/CardioIA/data/processed')

# Carregar dados
X_train = np.load(str(data_dir / 'X_train.npy'))
X_val = np.load(str(data_dir / 'X_val.npy'))
X_test = np.load(str(data_dir / 'X_test.npy'))
y_train = np.load(str(data_dir / 'y_train.npy'))
y_val = np.load(str(data_dir / 'y_val.npy'))
y_test = np.load(str(data_dir / 'y_test.npy'))

print("\n✓ Dados carregados com sucesso!")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_val shape: {X_val.shape}")
print(f"  X_test shape: {X_test.shape}")

# Adicionar dimensão de canal (grayscale -> 1 canal)
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(f"\n✓ Dimensão de canal adicionada:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_val shape: {X_val.shape}")
print(f"  X_test shape: {X_test.shape}")

# Número de classes
num_classes = len(np.unique(y_train))
print(f"\n✓ Número de classes: {num_classes}")

# Converter labels para one-hot encoding
y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
y_val_onehot = keras.utils.to_categorical(y_val, num_classes)
y_test_onehot = keras.utils.to_categorical(y_test, num_classes)

print(f"✓ Labels convertidos para one-hot encoding")

# ============================================================================
# SEÇÃO 2: MODELO CNN SIMPLES (TREINADO DO ZERO)
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 2: MODELO CNN SIMPLES (TREINADO DO ZERO)")
print("=" * 80)

def criar_cnn_simples(input_shape, num_classes):
    """
    Cria um modelo CNN simples do zero.
    
    Arquitetura:
    - 2 blocos convolucionais com pooling
    - Camadas densas para classificação
    
    Args:
        input_shape: Forma das imagens de entrada
        num_classes: Número de classes
    
    Returns:
        Modelo Keras compilado
    """
    model = models.Sequential([
        # Bloco 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloco 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Camadas Densas
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Criar modelo
print("\n✓ Criando modelo CNN simples...")
cnn_model = criar_cnn_simples(input_shape=X_train.shape[1:], num_classes=num_classes)

# Compilar
cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Modelo compilado!")
print("\nArquitetura do Modelo CNN Simples:")
cnn_model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Treinar
print("\n✓ Iniciando treinamento do modelo CNN simples...")
print("  (Este processo pode levar alguns minutos)\n")

history_cnn = cnn_model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n✓ Treinamento concluído!")

# ============================================================================
# SEÇÃO 3: TRANSFER LEARNING COM VGG16
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 3: TRANSFER LEARNING COM VGG16")
print("=" * 80)

def criar_transfer_learning_vgg16(input_shape, num_classes, congelar_base=True):
    """
    Cria um modelo de Transfer Learning usando VGG16.
    
    Args:
        input_shape: Forma das imagens de entrada
        num_classes: Número de classes
        congelar_base: Se deve congelar os pesos da base pré-treinada
    
    Returns:
        Modelo Keras compilado
    """
    # Carregar modelo pré-treinado (sem as camadas de topo)
    base_model = VGG16(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False
    )
    
    # Congelar camadas da base
    if congelar_base:
        base_model.trainable = False
    
    # Criar novo modelo
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Preparar dados para VGG16 (requer 3 canais RGB)
print("\n✓ Preparando dados para VGG16 (convertendo para RGB)...")
X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_val_rgb = np.repeat(X_val, 3, axis=-1)
X_test_rgb = np.repeat(X_test, 3, axis=-1)

print(f"  X_train_rgb shape: {X_train_rgb.shape}")

# Criar modelo VGG16
print("\n✓ Criando modelo Transfer Learning com VGG16...")
vgg_model = criar_transfer_learning_vgg16(
    input_shape=X_train_rgb.shape[1:],
    num_classes=num_classes,
    congelar_base=True
)

# Compilar
vgg_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Modelo compilado!")
print("\nArquitetura do Modelo Transfer Learning (VGG16):")
vgg_model.summary()

# Treinar
print("\n✓ Iniciando treinamento do modelo VGG16...")
print("  (Este processo pode levar alguns minutos)\n")

history_vgg = vgg_model.fit(
    X_train_rgb, y_train_onehot,
    validation_data=(X_val_rgb, y_val_onehot),
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n✓ Treinamento concluído!")

# ============================================================================
# SEÇÃO 4: TRANSFER LEARNING COM RESNET50
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 4: TRANSFER LEARNING COM RESNET50")
print("=" * 80)

def criar_transfer_learning_resnet50(input_shape, num_classes, congelar_base=True):
    """
    Cria um modelo de Transfer Learning usando ResNet50.
    
    Args:
        input_shape: Forma das imagens de entrada
        num_classes: Número de classes
        congelar_base: Se deve congelar os pesos da base pré-treinada
    
    Returns:
        Modelo Keras compilado
    """
    # Carregar modelo pré-treinado
    base_model = ResNet50(
        weights='imagenet',
        input_shape=input_shape,
        include_top=False
    )
    
    # Congelar camadas da base
    if congelar_base:
        base_model.trainable = False
    
    # Criar novo modelo
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Criar modelo ResNet50
print("\n✓ Criando modelo Transfer Learning com ResNet50...")
resnet_model = criar_transfer_learning_resnet50(
    input_shape=X_train_rgb.shape[1:],
    num_classes=num_classes,
    congelar_base=True
)

# Compilar
resnet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Modelo compilado!")
print("\nArquitetura do Modelo Transfer Learning (ResNet50):")
resnet_model.summary()

# Treinar
print("\n✓ Iniciando treinamento do modelo ResNet50...")
print("  (Este processo pode levar alguns minutos)\n")

history_resnet = resnet_model.fit(
    X_train_rgb, y_train_onehot,
    validation_data=(X_val_rgb, y_val_onehot),
    epochs=50,
    batch_size=16,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n✓ Treinamento concluído!")

# ============================================================================
# SEÇÃO 5: AVALIAÇÃO DOS MODELOS
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 5: AVALIAÇÃO DOS MODELOS")
print("=" * 80)

def avaliar_modelo(model, X_test, y_test, y_test_onehot, nome_modelo, usar_rgb=False):
    """
    Avalia um modelo e retorna as métricas.
    
    Args:
        model: Modelo Keras treinado
        X_test: Dados de teste
        y_test: Labels de teste (não one-hot)
        y_test_onehot: Labels de teste (one-hot)
        nome_modelo: Nome do modelo para exibição
        usar_rgb: Se os dados são RGB
    
    Returns:
        Dicionário com métricas
    """
    print(f"\n✓ Avaliando {nome_modelo}...")
    
    # Predições
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Métricas
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n  Métricas de {nome_modelo}:")
    print(f"  • Acurácia:  {acuracia:.4f}")
    print(f"  • Precisão:  {precisao:.4f}")
    print(f"  • Recall:    {recall:.4f}")
    print(f"  • F1-Score:  {f1:.4f}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    
    # Relatório de classificação
    relatorio = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'nome': nome_modelo,
        'acuracia': acuracia,
        'precisao': precisao,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'matriz_confusao': cm,
        'relatorio': relatorio
    }

# Avaliar todos os modelos
resultados_cnn = avaliar_modelo(cnn_model, X_test, y_test, y_test_onehot, "CNN Simples")
resultados_vgg = avaliar_modelo(vgg_model, X_test_rgb, y_test, y_test_onehot, "VGG16 (Transfer Learning)", usar_rgb=True)
resultados_resnet = avaliar_modelo(resnet_model, X_test_rgb, y_test, y_test_onehot, "ResNet50 (Transfer Learning)", usar_rgb=True)

# ============================================================================
# SEÇÃO 6: VISUALIZAÇÃO DOS RESULTADOS
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 6: VISUALIZAÇÃO DOS RESULTADOS")
print("=" * 80)

# Comparação de métricas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparação de Métricas entre Modelos', fontsize=16, fontweight='bold')

modelos = [resultados_cnn, resultados_vgg, resultados_resnet]
nomes = [m['nome'] for m in modelos]
metricas_dict = {
    'Acurácia': [m['acuracia'] for m in modelos],
    'Precisão': [m['precisao'] for m in modelos],
    'Recall': [m['recall'] for m in modelos],
    'F1-Score': [m['f1_score'] for m in modelos]
}

colors = ['#3498db', '#e74c3c', '#f39c12']

for idx, (ax, (metrica, valores)) in enumerate(zip(axes.flatten(), metricas_dict.items())):
    bars = ax.bar(nomes, valores, color=colors)
    ax.set_ylabel(metrica, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title(f'{metrica} por Modelo')
    ax.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ubuntu/CardioIA/reports/04_comparacao_metricas.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualização salva: 04_comparacao_metricas.png")
plt.close()

# Matrizes de Confusão
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Matrizes de Confusão', fontsize=14, fontweight='bold')

class_names = ['Normal', 'Cardiomegalia', 'Outras Patologias']

for ax, resultado in zip(axes, modelos):
    cm = resultado['matriz_confusao']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Contagem'})
    ax.set_title(resultado['nome'])
    ax.set_ylabel('Verdadeiro')
    ax.set_xlabel('Predito')

plt.tight_layout()
plt.savefig('/home/ubuntu/CardioIA/reports/05_matrizes_confusao.png', dpi=150, bbox_inches='tight')
print("✓ Visualização salva: 05_matrizes_confusao.png")
plt.close()

# Histórico de Treinamento
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Histórico de Treinamento', fontsize=14, fontweight='bold')

historicos = [
    (history_cnn, 'CNN Simples'),
    (history_vgg, 'VGG16'),
    (history_resnet, 'ResNet50')
]

for ax, (history, nome) in zip(axes, historicos):
    ax.plot(history.history['accuracy'], label='Acurácia Treino', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Acurácia Validação', linewidth=2)
    ax.set_title(f'{nome}')
    ax.set_xlabel('Época')
    ax.set_ylabel('Acurácia')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/CardioIA/reports/06_historico_treinamento.png', dpi=150, bbox_inches='tight')
print("✓ Visualização salva: 06_historico_treinamento.png")
plt.close()

# ============================================================================
# SEÇÃO 7: SALVAMENTO DOS MODELOS E RESULTADOS
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 7: SALVAMENTO DOS MODELOS E RESULTADOS")
print("=" * 80)

models_dir = Path('/home/ubuntu/CardioIA/models')
models_dir.mkdir(parents=True, exist_ok=True)

# Salvar modelos
print("\n✓ Salvando modelos treinados...")
cnn_model.save(str(models_dir / 'cnn_simples.h5'))
vgg_model.save(str(models_dir / 'vgg16_transfer_learning.h5'))
resnet_model.save(str(models_dir / 'resnet50_transfer_learning.h5'))
print("  Modelos salvos com sucesso!")

# Salvar resultados em JSON
resultados_json = {
    'data_execucao': datetime.now().isoformat(),
    'modelos': {
        'CNN Simples': {
            'acuracia': float(resultados_cnn['acuracia']),
            'precisao': float(resultados_cnn['precisao']),
            'recall': float(resultados_cnn['recall']),
            'f1_score': float(resultados_cnn['f1_score'])
        },
        'VGG16': {
            'acuracia': float(resultados_vgg['acuracia']),
            'precisao': float(resultados_vgg['precisao']),
            'recall': float(resultados_vgg['recall']),
            'f1_score': float(resultados_vgg['f1_score'])
        },
        'ResNet50': {
            'acuracia': float(resultados_resnet['acuracia']),
            'precisao': float(resultados_resnet['precisao']),
            'recall': float(resultados_resnet['recall']),
            'f1_score': float(resultados_resnet['f1_score'])
        }
    }
}

with open('/home/ubuntu/CardioIA/data/resultados_modelos.json', 'w', encoding='utf-8') as f:
    json.dump(resultados_json, f, indent=2, ensure_ascii=False)

print("✓ Resultados salvos em: resultados_modelos.json")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("RESUMO FINAL - PARTE 2")
print("=" * 80)

melhor_modelo = max(modelos, key=lambda x: x['acuracia'])

print(f"""
✓ TREINAMENTO E AVALIAÇÃO CONCLUÍDOS COM SUCESSO!

📊 Resultados dos Modelos:

1. CNN Simples:
   • Acurácia:  {resultados_cnn['acuracia']:.4f}
   • Precisão:  {resultados_cnn['precisao']:.4f}
   • Recall:    {resultados_cnn['recall']:.4f}
   • F1-Score:  {resultados_cnn['f1_score']:.4f}

2. VGG16 (Transfer Learning):
   • Acurácia:  {resultados_vgg['acuracia']:.4f}
   • Precisão:  {resultados_vgg['precisao']:.4f}
   • Recall:    {resultados_vgg['recall']:.4f}
   • F1-Score:  {resultados_vgg['f1_score']:.4f}

3. ResNet50 (Transfer Learning):
   • Acurácia:  {resultados_resnet['acuracia']:.4f}
   • Precisão:  {resultados_resnet['precisao']:.4f}
   • Recall:    {resultados_resnet['recall']:.4f}
   • F1-Score:  {resultados_resnet['f1_score']:.4f}

🏆 Melhor Modelo: {melhor_modelo['nome']} (Acurácia: {melhor_modelo['acuracia']:.4f})

💾 Arquivos Salvos:
   • cnn_simples.h5
   • vgg16_transfer_learning.h5
   • resnet50_transfer_learning.h5
   • resultados_modelos.json

📊 Relatórios Visuais:
   • 04_comparacao_metricas.png
   • 05_matrizes_confusao.png
   • 06_historico_treinamento.png

✅ Próximo passo: Criar protótipo de interface interativa (PARTE 3)
""")

print("=" * 80)
print("Fim da PARTE 2")
print("=" * 80)
