"""
PARTE 1: Pré-processamento e Organização de Imagens Médicas
CardioIA - A Nova Era da Cardiologia Inteligente

Este notebook implementa o pipeline completo de pré-processamento de imagens médicas
para classificação com CNN, incluindo:
- Download e exploração do dataset
- Redimensionamento e normalização
- Conversão de formatos
- Criação de conjuntos treino, validação e teste
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import shutil
from pathlib import Path
import json
from datetime import datetime

# Configurações
SEED = 42
np.random.seed(SEED)

print("=" * 80)
print("PARTE 1: PRÉ-PROCESSAMENTO E ORGANIZAÇÃO DE IMAGENS MÉDICAS")
print("CardioIA - A Nova Era da Cardiologia Inteligente")
print("=" * 80)
print(f"\nData de execução: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print(f"Seed para reprodutibilidade: {SEED}\n")

# ============================================================================
# SEÇÃO 1: DOWNLOAD E EXPLORAÇÃO DO DATASET
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 1: DOWNLOAD E EXPLORAÇÃO DO DATASET")
print("=" * 80)

# Para fins de demonstração, vamos criar um dataset sintético que simula
# radiografias de tórax com classificações cardíacas
# Em produção, você baixaria de: https://www.kaggle.com/datasets/

def criar_dataset_sintetico(num_amostras=200, tamanho_imagem=(224, 224)):
    """
    Cria um dataset sintético de imagens médicas para demonstração.
    Em um cenário real, isso seria substituído pelo download do dataset real.
    
    Args:
        num_amostras: Número total de imagens a gerar
        tamanho_imagem: Dimensões das imagens (altura, largura)
    
    Returns:
        Dicionário com informações das imagens e labels
    """
    print(f"\n✓ Criando dataset sintético com {num_amostras} imagens...")
    print(f"  Dimensões: {tamanho_imagem[0]}x{tamanho_imagem[1]} pixels")
    
    dataset_info = {
        'imagens': [],
        'labels': [],
        'caminhos': [],
        'descricoes': []
    }
    
    # Classes: 0 = Normal, 1 = Cardiomegalia, 2 = Outras Patologias
    classes = {
        0: 'Normal',
        1: 'Cardiomegalia',
        2: 'Outras_Patologias'
    }
    
    data_dir = Path('/home/ubuntu/CardioIA/data/raw_images')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Criar subdiretórios para cada classe
    for class_id, class_name in classes.items():
        class_dir = data_dir / class_name
        class_dir.mkdir(exist_ok=True)
    
    # Gerar imagens sintéticas
    amostras_por_classe = num_amostras // len(classes)
    
    for class_id, class_name in classes.items():
        for i in range(amostras_por_classe):
            # Gerar imagem sintética (simulando radiografia)
            if class_id == 0:  # Normal
                # Imagem com padrão mais uniforme
                img = np.random.normal(100, 20, tamanho_imagem).astype(np.uint8)
            elif class_id == 1:  # Cardiomegalia
                # Imagem com área cardíaca aumentada (mais branca no centro)
                img = np.random.normal(80, 25, tamanho_imagem).astype(np.uint8)
                y, x = np.ogrid[:tamanho_imagem[0], :tamanho_imagem[1]]
                mask = (x - tamanho_imagem[1]//2)**2 + (y - tamanho_imagem[0]//2)**2 <= (tamanho_imagem[0]//3)**2
                img[mask] = np.clip(img[mask] + 50, 0, 255).astype(np.uint8)
            else:  # Outras patologias
                # Imagem com padrão irregular
                img = np.random.normal(90, 30, tamanho_imagem).astype(np.uint8)
                img = cv2.GaussianBlur(img, (5, 5), 0)
            
            # Salvar imagem
            img_path = data_dir / class_name / f'{class_name}_{i:04d}.png'
            Image.fromarray(img).save(str(img_path))
            
            dataset_info['imagens'].append(img)
            dataset_info['labels'].append(class_id)
            dataset_info['caminhos'].append(str(img_path))
            dataset_info['descricoes'].append(f'{class_name} - Amostra {i}')
    
    print(f"✓ Dataset sintético criado com sucesso!")
    print(f"  Total de imagens: {len(dataset_info['imagens'])}")
    for class_id, class_name in classes.items():
        count = sum(1 for l in dataset_info['labels'] if l == class_id)
        print(f"  - {class_name}: {count} imagens")
    
    return dataset_info, classes

# Criar dataset
dataset_info, classes = criar_dataset_sintetico(num_amostras=200, tamanho_imagem=(224, 224))

# ============================================================================
# SEÇÃO 2: EXPLORAÇÃO E VISUALIZAÇÃO DO DATASET
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 2: EXPLORAÇÃO E VISUALIZAÇÃO DO DATASET")
print("=" * 80)

# Estatísticas do dataset
print("\n✓ Estatísticas do Dataset:")
print(f"  Total de imagens: {len(dataset_info['imagens'])}")
print(f"  Número de classes: {len(classes)}")
print(f"  Dimensões das imagens: {dataset_info['imagens'][0].shape}")
print(f"  Tipo de dados: {dataset_info['imagens'][0].dtype}")

# Distribuição de classes
print("\n✓ Distribuição de Classes:")
class_counts = pd.Series(dataset_info['labels']).value_counts().sort_index()
for class_id, count in class_counts.items():
    percentage = (count / len(dataset_info['labels'])) * 100
    print(f"  {classes[class_id]}: {count} ({percentage:.1f}%)")

# Visualizar amostras
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Amostras do Dataset - Pré-processamento', fontsize=14, fontweight='bold')

for idx, (ax, class_id) in enumerate(zip(axes.flatten(), [0, 0, 1, 1, 2, 2])):
    # Encontrar primeira imagem de cada classe
    img_idx = dataset_info['labels'].index(class_id)
    img = dataset_info['imagens'][img_idx]
    
    ax.imshow(img, cmap='gray')
    ax.set_title(f'{classes[class_id]}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('/home/ubuntu/CardioIA/reports/01_amostras_dataset.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualização salva: 01_amostras_dataset.png")
plt.close()

# ============================================================================
# SEÇÃO 3: PRÉ-PROCESSAMENTO DE IMAGENS
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 3: PRÉ-PROCESSAMENTO DE IMAGENS")
print("=" * 80)

class PreprocessadorImagens:
    """
    Classe para centralizar todas as operações de pré-processamento de imagens.
    """
    
    def __init__(self, tamanho_alvo=(224, 224)):
        """
        Inicializa o preprocessador.
        
        Args:
            tamanho_alvo: Dimensões alvo para redimensionamento
        """
        self.tamanho_alvo = tamanho_alvo
        self.historico = []
    
    def redimensionar(self, imagem):
        """
        Redimensiona a imagem para o tamanho alvo usando interpolação cúbica.
        
        Args:
            imagem: Array numpy da imagem
        
        Returns:
            Imagem redimensionada
        """
        img_redimensionada = cv2.resize(imagem, self.tamanho_alvo, interpolation=cv2.INTER_CUBIC)
        return img_redimensionada
    
    def normalizar(self, imagem):
        """
        Normaliza a imagem para o intervalo [0, 1].
        
        Args:
            imagem: Array numpy da imagem
        
        Returns:
            Imagem normalizada
        """
        img_normalizada = imagem.astype(np.float32) / 255.0
        return img_normalizada
    
    def padronizar(self, imagem, media=0.5, desvio=0.2):
        """
        Padroniza a imagem (z-score normalization).
        
        Args:
            imagem: Array numpy da imagem (normalizada entre 0 e 1)
            media: Média para padronização
            desvio: Desvio padrão para padronização
        
        Returns:
            Imagem padronizada
        """
        img_padronizada = (imagem - media) / desvio
        return img_padronizada
    
    def aplicar_histogram_equalization(self, imagem):
        """
        Aplica equalização de histograma para melhorar contraste.
        
        Args:
            imagem: Array numpy da imagem (valores 0-255)
        
        Returns:
            Imagem com contraste aprimorado
        """
        img_uint8 = (imagem * 255).astype(np.uint8) if imagem.max() <= 1 else imagem.astype(np.uint8)
        img_equalizado = cv2.equalizeHist(img_uint8)
        return img_equalizado.astype(np.float32) / 255.0
    
    def pipeline_completo(self, imagem, aplicar_equalizacao=True):
        """
        Aplica o pipeline completo de pré-processamento.
        
        Args:
            imagem: Array numpy da imagem
            aplicar_equalizacao: Se deve aplicar equalização de histograma
        
        Returns:
            Imagem pré-processada
        """
        # 1. Redimensionar
        img = self.redimensionar(imagem)
        
        # 2. Aplicar equalização (opcional)
        if aplicar_equalizacao:
            img = self.aplicar_histogram_equalization(img)
        
        # 3. Normalizar
        img = self.normalizar(img)
        
        # 4. Padronizar
        img = self.padronizar(img)
        
        return img

# Inicializar preprocessador
preprocessador = PreprocessadorImagens(tamanho_alvo=(224, 224))

print("\n✓ Técnicas de Pré-processamento Implementadas:")
print("  1. Redimensionamento (interpolação cúbica)")
print("  2. Equalização de Histograma (para melhorar contraste)")
print("  3. Normalização (escala 0-1)")
print("  4. Padronização (z-score normalization)")

# Aplicar pré-processamento a todas as imagens
print("\n✓ Aplicando pré-processamento a todas as imagens...")
imagens_processadas = []
for idx, img in enumerate(dataset_info['imagens']):
    img_processada = preprocessador.pipeline_completo(img)
    imagens_processadas.append(img_processada)
    if (idx + 1) % 50 == 0:
        print(f"  Processadas {idx + 1}/{len(dataset_info['imagens'])} imagens")

print(f"✓ Pré-processamento concluído!")
print(f"  Forma das imagens processadas: {imagens_processadas[0].shape}")
print(f"  Tipo de dados: {imagens_processadas[0].dtype}")
print(f"  Intervalo de valores: [{imagens_processadas[0].min():.3f}, {imagens_processadas[0].max():.3f}]")

# Visualizar antes e depois
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle('Comparação: Antes e Depois do Pré-processamento', fontsize=14, fontweight='bold')

for row, class_id in enumerate([0, 1, 2]):
    # Encontrar primeira imagem de cada classe
    img_idx = dataset_info['labels'].index(class_id)
    
    # Antes
    img_original = dataset_info['imagens'][img_idx]
    axes[row, 0].imshow(img_original, cmap='gray')
    axes[row, 0].set_title(f'{classes[class_id]} - Original')
    axes[row, 0].axis('off')
    
    # Redimensionado
    img_redim = preprocessador.redimensionar(img_original)
    axes[row, 1].imshow(img_redim, cmap='gray')
    axes[row, 1].set_title('Redimensionado')
    axes[row, 1].axis('off')
    
    # Com Equalização
    img_eq = preprocessador.aplicar_histogram_equalization(img_redim)
    axes[row, 2].imshow(img_eq, cmap='gray')
    axes[row, 2].set_title('Com Equalização')
    axes[row, 2].axis('off')
    
    # Processado Completo
    img_processada = imagens_processadas[img_idx]
    axes[row, 3].imshow(img_processada, cmap='gray')
    axes[row, 3].set_title('Processado Completo')
    axes[row, 3].axis('off')

plt.tight_layout()
plt.savefig('/home/ubuntu/CardioIA/reports/02_antes_depois_preprocessamento.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualização salva: 02_antes_depois_preprocessamento.png")
plt.close()

# ============================================================================
# SEÇÃO 4: CRIAÇÃO DE CONJUNTOS TREINO, VALIDAÇÃO E TESTE
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 4: CRIAÇÃO DE CONJUNTOS TREINO, VALIDAÇÃO E TESTE")
print("=" * 80)

# Converter para arrays numpy
X = np.array(imagens_processadas)
y = np.array(dataset_info['labels'])

print(f"\n✓ Dados preparados para divisão:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

# Primeira divisão: 70% treino+validação, 30% teste
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)

# Segunda divisão: 70% treino, 30% validação (do conjunto temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.30, random_state=SEED, stratify=y_temp
)

print(f"\n✓ Divisão dos Dados:")
print(f"  Treino:     {len(X_train)} imagens ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validação:  {len(X_val)} imagens ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Teste:      {len(X_test)} imagens ({len(X_test)/len(X)*100:.1f}%)")

# Verificar distribuição de classes em cada conjunto
print(f"\n✓ Distribuição de Classes por Conjunto:")
for conjunto_nome, y_conjunto in [('Treino', y_train), ('Validação', y_val), ('Teste', y_test)]:
    print(f"\n  {conjunto_nome}:")
    for class_id in range(len(classes)):
        count = np.sum(y_conjunto == class_id)
        percentage = (count / len(y_conjunto)) * 100
        print(f"    {classes[class_id]}: {count} ({percentage:.1f}%)")

# Salvar conjuntos em arquivos
print(f"\n✓ Salvando conjuntos de dados...")

data_processed_dir = Path('/home/ubuntu/CardioIA/data/processed')
data_processed_dir.mkdir(parents=True, exist_ok=True)

np.save(str(data_processed_dir / 'X_train.npy'), X_train)
np.save(str(data_processed_dir / 'X_val.npy'), X_val)
np.save(str(data_processed_dir / 'X_test.npy'), X_test)
np.save(str(data_processed_dir / 'y_train.npy'), y_train)
np.save(str(data_processed_dir / 'y_val.npy'), y_val)
np.save(str(data_processed_dir / 'y_test.npy'), y_test)

print(f"✓ Arquivos salvos em: {data_processed_dir}")

# Visualizar distribuição
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Distribuição de Classes nos Conjuntos de Dados', fontsize=14, fontweight='bold')

for ax, (conjunto_nome, y_conjunto) in zip(axes, [('Treino', y_train), ('Validação', y_val), ('Teste', y_test)]):
    class_counts = pd.Series(y_conjunto).value_counts().sort_index()
    class_names = [classes[i] for i in class_counts.index]
    
    bars = ax.bar(class_names, class_counts.values, color=['#3498db', '#e74c3c', '#f39c12'])
    ax.set_title(f'{conjunto_nome} (n={len(y_conjunto)})')
    ax.set_ylabel('Número de Imagens')
    ax.set_ylim(0, max(class_counts.values) * 1.2)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ubuntu/CardioIA/reports/03_distribuicao_conjuntos.png', dpi=150, bbox_inches='tight')
print("✓ Visualização salva: 03_distribuicao_conjuntos.png")
plt.close()

# ============================================================================
# SEÇÃO 5: DOCUMENTAÇÃO DO PIPELINE
# ============================================================================
print("\n" + "=" * 80)
print("SEÇÃO 5: DOCUMENTAÇÃO DO PIPELINE")
print("=" * 80)

pipeline_info = {
    'data_execucao': datetime.now().isoformat(),
    'seed': SEED,
    'dataset': {
        'total_imagens': len(X),
        'dimensoes': list(X.shape),
        'classes': classes,
        'distribuicao': {
            'treino': {
                'total': int(len(X_train)),
                'percentual': float(len(X_train)/len(X)*100)
            },
            'validacao': {
                'total': int(len(X_val)),
                'percentual': float(len(X_val)/len(X)*100)
            },
            'teste': {
                'total': int(len(X_test)),
                'percentual': float(len(X_test)/len(X)*100)
            }
        }
    },
    'preprocessamento': {
        'tecnicas_aplicadas': [
            'Redimensionamento (interpolação cúbica)',
            'Equalização de Histograma',
            'Normalização (0-1)',
            'Padronização (z-score)'
        ],
        'tamanho_alvo': list(preprocessador.tamanho_alvo),
        'intervalo_valores': [float(X.min()), float(X.max())]
    }
}

# Salvar como JSON
with open('/home/ubuntu/CardioIA/data/processed/pipeline_info.json', 'w', encoding='utf-8') as f:
    json.dump(pipeline_info, f, indent=2, ensure_ascii=False)

print("\n✓ Informações do Pipeline salvas em: pipeline_info.json")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "=" * 80)
print("RESUMO FINAL - PARTE 1")
print("=" * 80)

print(f"""
✓ PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!

📊 Estatísticas do Dataset:
   • Total de imagens: {len(X)}
   • Dimensões: {X.shape[1]}x{X.shape[2]} pixels
   • Número de canais: {X.shape[3] if len(X.shape) > 3 else 1}
   • Número de classes: {len(classes)}

📁 Divisão dos Dados:
   • Treino:     {len(X_train)} imagens ({len(X_train)/len(X)*100:.1f}%)
   • Validação:  {len(X_val)} imagens ({len(X_val)/len(X)*100:.1f}%)
   • Teste:      {len(X_test)} imagens ({len(X_test)/len(X)*100:.1f}%)

🔧 Técnicas de Pré-processamento Aplicadas:
   1. Redimensionamento (interpolação cúbica)
   2. Equalização de Histograma
   3. Normalização (escala 0-1)
   4. Padronização (z-score normalization)

📈 Intervalo de Valores dos Dados:
   • Mínimo: {X.min():.4f}
   • Máximo: {X.max():.4f}
   • Média: {X.mean():.4f}
   • Desvio Padrão: {X.std():.4f}

💾 Arquivos Gerados:
   • X_train.npy, y_train.npy
   • X_val.npy, y_val.npy
   • X_test.npy, y_test.npy
   • pipeline_info.json

📊 Relatórios Visuais:
   • 01_amostras_dataset.png
   • 02_antes_depois_preprocessamento.png
   • 03_distribuicao_conjuntos.png

✅ Próximo passo: Implementar modelos CNN (PARTE 2)
""")

print("=" * 80)
print("Fim da PARTE 1")
print("=" * 80)
