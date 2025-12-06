#!/usr/bin/env python3
"""
PARTE 2 OTIMIZADA: Classificação de Imagens Médicas com CNN
CardioIA - A Nova Era da Cardiologia Inteligente
Versão simplificada para execução rápida
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("="*80)
print("PARTE 2: CLASSIFICAÇÃO DE IMAGENS MÉDICAS COM CNN (VERSÃO OTIMIZADA)")
print("="*80)
print(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")

# ============================================================================
# CARREGAR DADOS
# ============================================================================
print("✓ Carregando dados pré-processados...")
data_dir = Path('/home/ubuntu/CardioIA/data/processed')

X_train = np.load(str(data_dir / 'X_train.npy'))
X_val = np.load(str(data_dir / 'X_val.npy'))
X_test = np.load(str(data_dir / 'X_test.npy'))
y_train = np.load(str(data_dir / 'y_train.npy'))
y_val = np.load(str(data_dir / 'y_val.npy'))
y_test = np.load(str(data_dir / 'y_test.npy'))

# Adicionar dimensão de canal
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

num_classes = len(np.unique(y_train))
y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
y_val_onehot = keras.utils.to_categorical(y_val, num_classes)
y_test_onehot = keras.utils.to_categorical(y_test, num_classes)

print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print(f"  Número de classes: {num_classes}\n")

# Preparar dados RGB
X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_val_rgb = np.repeat(X_val, 3, axis=-1)
X_test_rgb = np.repeat(X_test, 3, axis=-1)

# ============================================================================
# MODELO 1: CNN SIMPLES
# ============================================================================
print("✓ Modelo 1: CNN Simples")
print("  Arquitetura: 2 blocos conv + Dense layers")

cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("  Treinando...")
history_cnn = cnn_model.fit(
    X_train, y_train_onehot,
    validation_data=(X_val, y_val_onehot),
    epochs=20,
    batch_size=16,
    verbose=0
)

y_pred_cnn = np.argmax(cnn_model.predict(X_test, verbose=0), axis=1)
cm_cnn = confusion_matrix(y_test, y_pred_cnn)
acc_cnn = accuracy_score(y_test, y_pred_cnn)
prec_cnn = precision_score(y_test, y_pred_cnn, average='weighted', zero_division=0)
rec_cnn = recall_score(y_test, y_pred_cnn, average='weighted', zero_division=0)
f1_cnn = f1_score(y_test, y_pred_cnn, average='weighted', zero_division=0)

print(f"  Resultados: Acurácia={acc_cnn:.4f}, F1={f1_cnn:.4f}\n")

# ============================================================================
# MODELO 2: VGG16 TRANSFER LEARNING
# ============================================================================
print("✓ Modelo 2: VGG16 (Transfer Learning)")
print("  Arquitetura: VGG16 pré-treinado + camadas customizadas")

base_vgg = VGG16(weights='imagenet', input_shape=X_train_rgb.shape[1:], include_top=False)
base_vgg.trainable = False

vgg_model = models.Sequential([
    base_vgg,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

vgg_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print("  Treinando...")
history_vgg = vgg_model.fit(
    X_train_rgb, y_train_onehot,
    validation_data=(X_val_rgb, y_val_onehot),
    epochs=20,
    batch_size=16,
    verbose=0
)

y_pred_vgg = np.argmax(vgg_model.predict(X_test_rgb, verbose=0), axis=1)
cm_vgg = confusion_matrix(y_test, y_pred_vgg)
acc_vgg = accuracy_score(y_test, y_pred_vgg)
prec_vgg = precision_score(y_test, y_pred_vgg, average='weighted', zero_division=0)
rec_vgg = recall_score(y_test, y_pred_vgg, average='weighted', zero_division=0)
f1_vgg = f1_score(y_test, y_pred_vgg, average='weighted', zero_division=0)

print(f"  Resultados: Acurácia={acc_vgg:.4f}, F1={f1_vgg:.4f}\n")

# ============================================================================
# MODELO 3: RESNET50 TRANSFER LEARNING
# ============================================================================
print("✓ Modelo 3: ResNet50 (Transfer Learning)")
print("  Arquitetura: ResNet50 pré-treinado + camadas customizadas")

base_resnet = ResNet50(weights='imagenet', input_shape=X_train_rgb.shape[1:], include_top=False)
base_resnet.trainable = False

resnet_model = models.Sequential([
    base_resnet,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

resnet_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print("  Treinando...")
history_resnet = resnet_model.fit(
    X_train_rgb, y_train_onehot,
    validation_data=(X_val_rgb, y_val_onehot),
    epochs=20,
    batch_size=16,
    verbose=0
)

y_pred_resnet = np.argmax(resnet_model.predict(X_test_rgb, verbose=0), axis=1)
cm_resnet = confusion_matrix(y_test, y_pred_resnet)
acc_resnet = accuracy_score(y_test, y_pred_resnet)
prec_resnet = precision_score(y_test, y_pred_resnet, average='weighted', zero_division=0)
rec_resnet = recall_score(y_test, y_pred_resnet, average='weighted', zero_division=0)
f1_resnet = f1_score(y_test, y_pred_resnet, average='weighted', zero_division=0)

print(f"  Resultados: Acurácia={acc_resnet:.4f}, F1={f1_resnet:.4f}\n")

# ============================================================================
# VISUALIZAÇÕES
# ============================================================================
print("✓ Gerando visualizações...")

# 1. Comparação de Métricas
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparação de Métricas entre Modelos', fontsize=16, fontweight='bold')

modelos = ['CNN Simples', 'VGG16', 'ResNet50']
metricas = {
    'Acurácia': [acc_cnn, acc_vgg, acc_resnet],
    'Precisão': [prec_cnn, prec_vgg, prec_resnet],
    'Recall': [rec_cnn, rec_vgg, rec_resnet],
    'F1-Score': [f1_cnn, f1_vgg, f1_resnet]
}

colors = ['#3498db', '#e74c3c', '#f39c12']

for idx, (ax, (metrica, valores)) in enumerate(zip(axes.flatten(), metricas.items())):
    bars = ax.bar(modelos, valores, color=colors)
    ax.set_ylabel(metrica, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title(f'{metrica} por Modelo')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('/home/ubuntu/CardioIA/reports/04_comparacao_metricas.png', dpi=150, bbox_inches='tight')
print("  ✓ 04_comparacao_metricas.png")
plt.close()

# 2. Matrizes de Confusão
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Matrizes de Confusão', fontsize=14, fontweight='bold')

class_names = ['Normal', 'Cardiomegalia', 'Outras Patologias']

for ax, (cm, nome) in zip(axes, [(cm_cnn, 'CNN Simples'), (cm_vgg, 'VGG16'), (cm_resnet, 'ResNet50')]):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Contagem'})
    ax.set_title(nome)
    ax.set_ylabel('Verdadeiro')
    ax.set_xlabel('Predito')

plt.tight_layout()
plt.savefig('/home/ubuntu/CardioIA/reports/05_matrizes_confusao.png', dpi=150, bbox_inches='tight')
print("  ✓ 05_matrizes_confusao.png")
plt.close()

# 3. Histórico de Treinamento
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Histórico de Treinamento', fontsize=14, fontweight='bold')

for ax, (history, nome) in zip(axes, [(history_cnn, 'CNN Simples'), (history_vgg, 'VGG16'), (history_resnet, 'ResNet50')]):
    ax.plot(history.history['accuracy'], label='Acurácia Treino', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Acurácia Validação', linewidth=2)
    ax.set_title(f'{nome}')
    ax.set_xlabel('Época')
    ax.set_ylabel('Acurácia')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/CardioIA/reports/06_historico_treinamento.png', dpi=150, bbox_inches='tight')
print("  ✓ 06_historico_treinamento.png")
plt.close()

# ============================================================================
# SALVAR MODELOS E RESULTADOS
# ============================================================================
print("\n✓ Salvando modelos e resultados...")

models_dir = Path('/home/ubuntu/CardioIA/models')
models_dir.mkdir(parents=True, exist_ok=True)

cnn_model.save(str(models_dir / 'cnn_simples.h5'))
vgg_model.save(str(models_dir / 'vgg16_transfer_learning.h5'))
resnet_model.save(str(models_dir / 'resnet50_transfer_learning.h5'))

# Salvar resultados em JSON
resultados_json = {
    'data_execucao': datetime.now().isoformat(),
    'modelos': {
        'CNN Simples': {
            'acuracia': float(acc_cnn),
            'precisao': float(prec_cnn),
            'recall': float(rec_cnn),
            'f1_score': float(f1_cnn)
        },
        'VGG16': {
            'acuracia': float(acc_vgg),
            'precisao': float(prec_vgg),
            'recall': float(rec_vgg),
            'f1_score': float(f1_vgg)
        },
        'ResNet50': {
            'acuracia': float(acc_resnet),
            'precisao': float(prec_resnet),
            'recall': float(rec_resnet),
            'f1_score': float(f1_resnet)
        }
    }
}

with open('/home/ubuntu/CardioIA/data/resultados_modelos.json', 'w', encoding='utf-8') as f:
    json.dump(resultados_json, f, indent=2, ensure_ascii=False)

print("  ✓ Modelos salvos em: /models/")
print("  ✓ Resultados salvos em: /data/resultados_modelos.json")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMO FINAL - PARTE 2")
print("="*80)

print(f"\n📊 Resultados dos Modelos:\n")
print(f"1. CNN Simples:")
print(f"   • Acurácia:  {acc_cnn:.4f}")
print(f"   • Precisão:  {prec_cnn:.4f}")
print(f"   • Recall:    {rec_cnn:.4f}")
print(f"   • F1-Score:  {f1_cnn:.4f}\n")

print(f"2. VGG16 (Transfer Learning):")
print(f"   • Acurácia:  {acc_vgg:.4f}")
print(f"   • Precisão:  {prec_vgg:.4f}")
print(f"   • Recall:    {rec_vgg:.4f}")
print(f"   • F1-Score:  {f1_vgg:.4f}\n")

print(f"3. ResNet50 (Transfer Learning):")
print(f"   • Acurácia:  {acc_resnet:.4f}")
print(f"   • Precisão:  {prec_resnet:.4f}")
print(f"   • Recall:    {rec_resnet:.4f}")
print(f"   • F1-Score:  {f1_resnet:.4f}\n")

melhor = max([('CNN Simples', acc_cnn), ('VGG16', acc_vgg), ('ResNet50', acc_resnet)], key=lambda x: x[1])
print(f"🏆 Melhor Modelo: {melhor[0]} (Acurácia: {melhor[1]:.4f})")

print("\n💾 Arquivos Gerados:")
print("   • cnn_simples.h5")
print("   • vgg16_transfer_learning.h5")
print("   • resnet50_transfer_learning.h5")
print("   • 04_comparacao_metricas.png")
print("   • 05_matrizes_confusao.png")
print("   • 06_historico_treinamento.png")

print("\n✅ PARTE 2 CONCLUÍDA COM SUCESSO!")
print("="*80)
