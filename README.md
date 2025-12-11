# 🔬 Sistema de Classificação de Câncer de Pele com Deep Learning

## 📋 Visão Geral

Este sistema utiliza **Deep Learning** para classificar lesões de pele em 7 categorias diferentes, auxiliando na detecção precoce de câncer de pele. O modelo é baseado em **Transfer Learning** com a arquitetura **MobileNetV2** e inclui otimização de thresholds para melhorar a sensibilidade clínica.

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DE CLASSIFICAÇÃO                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐ │
│   │  Imagem  │ -> │ Preprocessa- │ -> │ MobileNetV2 │ -> │ Classifi- │ │
│   │ (224x224)│    │    mento     │    │   (Base)    │    │   cador   │ │
│   └──────────┘    └──────────────┘    └─────────────┘    └───────────┘ │
│                                                                │         │
│                                                                ▼         │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                    OTIMIZAÇÃO DE THRESHOLDS                       │  │
│   │                     (Youden's J Index)                            │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                │         │
│                                                                ▼         │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │                    PREDIÇÃO FINAL + INTERPRETAÇÃO                 │  │
│   │              (7 classes + Recomendação Clínica)                   │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Tipo de Modelo

| Característica | Classificação |
|----------------|---------------|
| **Machine Learning** | ✅ Sim |
| **Deep Learning** | ✅ Sim (Redes Neurais Profundas) |
| **Transfer Learning** | ✅ Sim (Pesos pré-treinados do ImageNet) |
| **Aprendizado Supervisionado** | ✅ Sim (Labels conhecidos) |
| **Reinforcement Learning** | ❌ Não |

### Por que é Machine Learning?
- O modelo **aprende padrões** automaticamente a partir dos dados
- Usa **otimização iterativa** (backpropagation + gradient descent)
- **Generaliza** para novas imagens não vistas durante o treinamento

### Por que é Deep Learning?
- Utiliza **rede neural profunda** (MobileNetV2 tem ~150+ camadas)
- **Extração automática de features** hierárquicas
- Não requer engenharia manual de características

---

## 🛠️ Tecnologias Utilizadas

### Frameworks e Bibliotecas

| Tecnologia | Versão | Propósito |
|------------|--------|-----------|
| **TensorFlow** | 2.x | Framework principal de Deep Learning |
| **Keras** | Integrado | API de alto nível para construção do modelo |
| **NumPy** | - | Operações numéricas e manipulação de arrays |
| **Pandas** | - | Manipulação de dados e metadados |
| **Scikit-learn** | - | Métricas, encoding e split de dados |
| **Matplotlib/Seaborn** | - | Visualizações e gráficos |

### Hardware Suportado
- ✅ GPU NVIDIA (CUDA) - Recomendado
- ✅ CPU - Funcional, porém mais lento

---

## 📊 Dataset: HAM10000

O modelo foi treinado no dataset **HAM10000** (Human Against Machine with 10,000 training images):

| Classe | Nome Completo | Categoria | Amostras |
|--------|---------------|-----------|----------|
| **nv** | Melanocytic Nevi | Benigno | ~6.700 (67%) |
| **mel** | Melanoma | **Maligno** ⚠️ | ~1.100 (11%) |
| **bkl** | Benign Keratosis | Benigno | ~1.100 (11%) |
| **bcc** | Basal Cell Carcinoma | **Maligno** | ~500 (5%) |
| **akiec** | Actinic Keratoses | Pré-canceroso | ~300 (3%) |
| **vasc** | Vascular Lesions | Benigno | ~140 (1.4%) |
| **df** | Dermatofibroma | Benigno | ~115 (1.1%) |

**Desafio Principal:** Desbalanceamento severo de classes (67% de uma única classe)

---

## 🏛️ Arquitetura do Modelo

### MobileNetV2 (Base)

**MobileNetV2** é uma arquitetura de rede neural convolucional otimizada para eficiência:

```
┌─────────────────────────────────────────────────────────────┐
│                      MobileNetV2                             │
├─────────────────────────────────────────────────────────────┤
│  • Desenvolvida pelo Google (2018)                          │
│  • ~3.4 milhões de parâmetros (leve e eficiente)            │
│  • Usa "Inverted Residuals" e "Linear Bottlenecks"          │
│  • Pré-treinada em ImageNet (1.4M imagens, 1000 classes)    │
│  • Ideal para aplicações móveis e tempo real                │
└─────────────────────────────────────────────────────────────┘
```

### Classificador Customizado (Head)

```
MobileNetV2 (congelado/fine-tuned)
         │
         ▼
┌─────────────────────────┐
│ Global Average Pooling  │  <- Reduz dimensionalidade espacial
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│    Dropout (0.3)        │  <- Regularização (evita overfitting)
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Dense (256, ReLU)      │  <- Camada densa com ativação ReLU
│  + L2 Regularization    │
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│    Dropout (0.2)        │  <- Regularização adicional
└─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Dense (7, Softmax)     │  <- Saída: probabilidades para 7 classes
└─────────────────────────┘
```

---

## 🎯 Estratégia de Treinamento

### Treinamento em 2 Fases

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FASE 1: Feature Extraction                   │
├─────────────────────────────────────────────────────────────────────┤
│  • Épocas: 10                                                        │
│  • Learning Rate: 1e-4                                               │
│  • Base MobileNetV2: CONGELADA (não treina)                         │
│  • Treina apenas: Camadas do classificador                          │
│  • Objetivo: Adaptar o head ao novo problema                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FASE 2: Fine-Tuning                          │
├─────────────────────────────────────────────────────────────────────┤
│  • Épocas: 20                                                        │
│  • Learning Rate: 1e-5 (10x menor)                                  │
│  • Base MobileNetV2: ÚLTIMAS 50 CAMADAS desbloqueadas               │
│  • Objetivo: Ajustar features específicas para lesões de pele       │
└─────────────────────────────────────────────────────────────────────┘
```

### Otimizador: Adam

O **Adam** (Adaptive Moment Estimation) é o otimizador utilizado:

- Combina as vantagens do **Momentum** e **RMSprop**
- Adapta a learning rate individualmente para cada parâmetro
- Converge rapidamente e é robusto a hiperparâmetros

```
θ(t+1) = θ(t) - α * m̂(t) / (√v̂(t) + ε)

Onde:
  - m̂: Estimativa do primeiro momento (média dos gradientes)
  - v̂: Estimativa do segundo momento (variância dos gradientes)
  - α: Learning rate
  - ε: Constante de estabilidade numérica
```

---

## ⚖️ Tratamento do Desbalanceamento de Classes

### 1. Focal Loss

A **Focal Loss** foca em exemplos difíceis de classificar:

```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

Onde:
  - p_t: Probabilidade do modelo para a classe correta
  - γ (gamma): Fator de foco (usamos γ=2.0)
  - α_t: Peso da classe
```

**Efeito:** Exemplos fáceis (alta probabilidade) contribuem menos para a loss, permitindo que o modelo foque em casos difíceis.

### 2. Class Weights

Pesos inversamente proporcionais à frequência de cada classe:

```
weight(classe) = n_samples / (n_classes * n_samples_classe)
```

Classes raras recebem pesos maiores, forçando o modelo a prestar mais atenção nelas.

---

## 📈 Otimização de Thresholds (Youden's J Index)

### Problema
Por padrão, classificadores usam threshold de **0.5** para todas as classes. Isso não é ideal para:
- Classes desbalanceadas
- Aplicações médicas onde falsos negativos são críticos

### Solução: Youden's J Index

```
J = Sensibilidade + Especificidade - 1
J = TPR - FPR

Onde:
  - TPR (True Positive Rate): Sensibilidade
  - FPR (False Positive Rate): 1 - Especificidade
```

O threshold ótimo é o ponto na curva ROC que **maximiza J** (mais distante da diagonal).

### Ajuste de Segurança para Melanoma

Para melanoma (câncer mais perigoso), aplicamos uma **margem de segurança**:

```
threshold_melanoma = threshold_youden - 0.10
```

**Efeito:** Maior sensibilidade (menos melanomas perdidos), aceitando mais falsos positivos.

---

## 🔄 Pipeline de Inferência

```python
def predict_image(image_path, model, thresholds):
    # 1. Carregar e redimensionar imagem
    img = load_image(image_path, size=(224, 224))
    
    # 2. Preprocessar para MobileNetV2
    img = preprocess_input(img)  # Normaliza para [-1, 1]
    
    # 3. Obter probabilidades
    probabilities = model.predict(img)
    
    # 4. Aplicar thresholds otimizados
    above_threshold = probabilities >= thresholds
    
    if any(above_threshold):
        # Escolher classe com maior probabilidade acima do threshold
        predicted_class = argmax(masked_probabilities)
    else:
        # Fallback: argmax padrão
        predicted_class = argmax(probabilities)
    
    # 5. Interpretação clínica
    if predicted_class in ['mel', 'bcc', 'akiec']:
        return "POTENCIALMENTE MALIGNO - Consulte um dermatologista"
    else:
        return "PROVAVELMENTE BENIGNO - Monitore mudanças"
```

---

## 📊 Resultados Esperados

### Métricas de Performance

| Métrica | Valor Típico |
|---------|--------------|
| **Accuracy** | ~75-80% |
| **Weighted F1-Score** | ~0.75-0.80 |
| **Melanoma Sensitivity** | ~70-85% |
| **AUC (média)** | ~0.85-0.92 |

### Curvas ROC

O modelo gera curvas ROC para cada classe, permitindo avaliar o trade-off entre sensibilidade e especificidade.

---

## 📁 Arquivos Gerados

| Arquivo | Descrição |
|---------|-----------|
| `skin_cancer_mobilenetv2_final.keras` | Modelo treinado completo |
| `optimized_thresholds.npy` | Thresholds otimizados por classe |
| `label_encoder.pkl` | Codificação de labels |
| `model_config.json` | Configurações e metadados |
| `training_curves.png` | Gráficos de treino |
| `roc_curves.png` | Curvas ROC por classe |
| `confusion_matrices.png` | Matrizes de confusão |

---

## 🚀 Como Usar para Inferência

```python
import tensorflow as tf
import numpy as np
import json

# Carregar modelo e configurações
model = tf.keras.models.load_model('skin_cancer_mobilenetv2_final.keras')
thresholds = np.load('optimized_thresholds.npy')

with open('model_config.json', 'r') as f:
    config = json.load(f)

# Fazer predição
result = predict_image(
    image_path='sua_imagem.jpg',
    model=model,
    thresholds=thresholds,
    class_names=config['class_names']
)

print(f"Classe: {result['predicted_class']}")
print(f"Confiança: {result['confidence']*100:.1f}%")
print(f"Recomendação: {result['recommendation']}")
```

---

## ⚠️ Limitações e Avisos

### Limitações Técnicas
- Dataset limitado a 10.000 imagens
- Imagens de dermoscopia (não funciona bem com fotos de celular)
- Desbalanceamento de classes pode afetar classes minoritárias

### Aviso Médico

> ⚠️ **IMPORTANTE**: Este modelo é para **fins educacionais e de pesquisa apenas**.
> 
> NÃO deve ser usado como substituto para diagnóstico médico profissional.
> Sempre consulte um dermatologista qualificado para avaliação de lesões de pele.

---

## 📚 Referências

1. **MobileNetV2**: Sandler, M., et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (2018)
2. **HAM10000**: Tschandl, P., et al. "The HAM10000 dataset" (2018)
3. **Focal Loss**: Lin, T., et al. "Focal Loss for Dense Object Detection" (2017)
4. **Youden's Index**: Youden, W.J. "Index for rating diagnostic tests" (1950)

---

## 👥 Autor

Desenvolvido como projeto acadêmico para classificação de lesões de pele usando técnicas de Deep Learning.

---

*Última atualização: Dezembro 2025*
