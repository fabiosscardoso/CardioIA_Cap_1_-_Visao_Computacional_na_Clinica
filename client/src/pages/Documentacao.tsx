import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { FileText, Image, Brain, BarChart3, Download } from "lucide-react";

export default function Documentacao() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <Link href="/">
            <Button variant="ghost">← Voltar</Button>
          </Link>
          <h1 className="text-xl font-bold">Documentação do Projeto</h1>
          <div className="w-24"></div>
        </div>
      </header>

      <div className="container max-w-5xl mx-auto px-4 py-8">
        <Tabs defaultValue="parte1" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="parte1">PARTE 1</TabsTrigger>
            <TabsTrigger value="parte2">PARTE 2</TabsTrigger>
            <TabsTrigger value="arquitetura">Arquitetura</TabsTrigger>
          </TabsList>

          {/* PARTE 1: Pré-processamento */}
          <TabsContent value="parte1" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Image className="w-6 h-6 text-blue-600" />
                  PARTE 1: Pré-processamento e Organização de Imagens
                </CardTitle>
                <CardDescription>
                  Pipeline completo de preparação de dados para classificação com CNN
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">1. Dataset</h3>
                  <p className="text-gray-600 mb-3">
                    Para este projeto acadêmico, foi criado um dataset sintético de 198 imagens simulando
                    radiografias de tórax, divididas em três classes:
                  </p>
                  <ul className="list-disc list-inside text-gray-600 space-y-1 ml-4">
                    <li><strong>Normal:</strong> 66 imagens (33.3%)</li>
                    <li><strong>Cardiomegalia:</strong> 66 imagens (33.3%)</li>
                    <li><strong>Outras Patologias:</strong> 66 imagens (33.3%)</li>
                  </ul>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">2. Técnicas de Pré-processamento</h3>
                  <div className="space-y-3">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-blue-900 mb-2">Redimensionamento</h4>
                      <p className="text-sm text-gray-700">
                        Todas as imagens foram redimensionadas para 224×224 pixels usando interpolação cúbica,
                        garantindo compatibilidade com os modelos de Transfer Learning.
                      </p>
                    </div>
                    <div className="bg-cyan-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-cyan-900 mb-2">Equalização de Histograma</h4>
                      <p className="text-sm text-gray-700">
                        Aplicada para melhorar o contraste das imagens médicas, facilitando a identificação
                        de características relevantes pelos modelos.
                      </p>
                    </div>
                    <div className="bg-purple-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-purple-900 mb-2">Normalização</h4>
                      <p className="text-sm text-gray-700">
                        Valores dos pixels normalizados para o intervalo [0, 1], acelerando a convergência
                        durante o treinamento.
                      </p>
                    </div>
                    <div className="bg-green-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-green-900 mb-2">Padronização (Z-score)</h4>
                      <p className="text-sm text-gray-700">
                        Aplicação de z-score normalization para centralizar os dados em torno da média zero
                        com desvio padrão unitário.
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">3. Divisão dos Dados</h3>
                  <div className="grid md:grid-cols-3 gap-4">
                    <Card className="bg-blue-50 border-blue-200">
                      <CardHeader>
                        <CardTitle className="text-blue-900">Treino</CardTitle>
                        <CardDescription className="text-blue-700">96 imagens (48.5%)</CardDescription>
                      </CardHeader>
                    </Card>
                    <Card className="bg-cyan-50 border-cyan-200">
                      <CardHeader>
                        <CardTitle className="text-cyan-900">Validação</CardTitle>
                        <CardDescription className="text-cyan-700">42 imagens (21.2%)</CardDescription>
                      </CardHeader>
                    </Card>
                    <Card className="bg-purple-50 border-purple-200">
                      <CardHeader>
                        <CardTitle className="text-purple-900">Teste</CardTitle>
                        <CardDescription className="text-purple-700">60 imagens (30.3%)</CardDescription>
                      </CardHeader>
                    </Card>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">4. Visualizações</h3>
                  <div className="space-y-4">
                    <div>
                      <h4 className="font-medium mb-2">Amostras do Dataset</h4>
                      <img
                        src="/reports/01_amostras_dataset.png"
                        alt="Amostras do Dataset"
                        className="w-full rounded-lg border"
                      />
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">Antes e Depois do Pré-processamento</h4>
                      <img
                        src="/reports/02_antes_depois_preprocessamento.png"
                        alt="Antes e Depois"
                        className="w-full rounded-lg border"
                      />
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">Distribuição dos Conjuntos</h4>
                      <img
                        src="/reports/03_distribuicao_conjuntos.png"
                        alt="Distribuição"
                        className="w-full rounded-lg border"
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* PARTE 2: Modelos CNN */}
          <TabsContent value="parte2" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-6 h-6 text-cyan-600" />
                  PARTE 2: Classificação com CNN e Transfer Learning
                </CardTitle>
                <CardDescription>
                  Implementação e avaliação de três modelos de classificação
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">1. Modelos Implementados</h3>
                  <div className="space-y-4">
                    <Card className="border-blue-200">
                      <CardHeader>
                        <CardTitle className="text-blue-900">CNN Simples</CardTitle>
                        <CardDescription>Modelo treinado do zero</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-gray-600 mb-3">
                          Arquitetura com 3 blocos convolucionais seguidos de camadas densas:
                        </p>
                        <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                          <li>2 camadas Conv2D (32 filtros) + MaxPooling + Dropout</li>
                          <li>2 camadas Conv2D (64 filtros) + MaxPooling + Dropout</li>
                          <li>Flatten + Dense (128) + Dropout + Dense (3 classes)</li>
                        </ul>
                        <div className="mt-3 p-3 bg-blue-50 rounded">
                          <p className="text-sm"><strong>Acurácia:</strong> 33.33%</p>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="border-cyan-200">
                      <CardHeader>
                        <CardTitle className="text-cyan-900">VGG16 (Transfer Learning)</CardTitle>
                        <CardDescription>Melhor desempenho</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-gray-600 mb-3">
                          Utiliza pesos pré-treinados do ImageNet com camadas customizadas:
                        </p>
                        <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                          <li>Base VGG16 (congelada)</li>
                          <li>GlobalAveragePooling2D</li>
                          <li>Dense (128) + Dropout + Dense (3 classes)</li>
                        </ul>
                        <div className="mt-3 p-3 bg-cyan-50 rounded">
                          <p className="text-sm"><strong>Acurácia:</strong> 40.00% 🏆</p>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="border-purple-200">
                      <CardHeader>
                        <CardTitle className="text-purple-900">ResNet50 (Transfer Learning)</CardTitle>
                        <CardDescription>Arquitetura residual</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-gray-600 mb-3">
                          Utiliza conexões residuais com pesos do ImageNet:
                        </p>
                        <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                          <li>Base ResNet50 (congelada)</li>
                          <li>GlobalAveragePooling2D</li>
                          <li>Dense (128) + Dropout + Dense (3 classes)</li>
                        </ul>
                        <div className="mt-3 p-3 bg-purple-50 rounded">
                          <p className="text-sm"><strong>Acurácia:</strong> 33.33%</p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">2. Métricas de Avaliação</h3>
                  <p className="text-gray-600 mb-3">
                    Todos os modelos foram avaliados usando as seguintes métricas:
                  </p>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-blue-900 mb-2">Acurácia</h4>
                      <p className="text-sm text-gray-700">
                        Proporção de predições corretas sobre o total de predições.
                      </p>
                    </div>
                    <div className="bg-cyan-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-cyan-900 mb-2">Precisão</h4>
                      <p className="text-sm text-gray-700">
                        Proporção de verdadeiros positivos sobre todos os positivos preditos.
                      </p>
                    </div>
                    <div className="bg-purple-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-purple-900 mb-2">Recall</h4>
                      <p className="text-sm text-gray-700">
                        Proporção de verdadeiros positivos sobre todos os positivos reais.
                      </p>
                    </div>
                    <div className="bg-green-50 p-4 rounded-lg">
                      <h4 className="font-semibold text-green-900 mb-2">F1-Score</h4>
                      <p className="text-sm text-gray-700">
                        Média harmônica entre precisão e recall.
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">3. Configuração de Treinamento</h3>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <ul className="space-y-2 text-sm text-gray-700">
                      <li><strong>Épocas:</strong> 20 (com early stopping)</li>
                      <li><strong>Batch Size:</strong> 16</li>
                      <li><strong>Otimizador:</strong> Adam (lr=0.001 para CNN, lr=0.0001 para Transfer Learning)</li>
                      <li><strong>Loss Function:</strong> Categorical Crossentropy</li>
                      <li><strong>Callbacks:</strong> EarlyStopping e ReduceLROnPlateau</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Arquitetura do Sistema */}
          <TabsContent value="arquitetura" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="w-6 h-6 text-purple-600" />
                  Arquitetura do Sistema
                </CardTitle>
                <CardDescription>
                  Estrutura técnica e tecnologias utilizadas
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Tecnologias Utilizadas</h3>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-semibold mb-2">Backend / ML</h4>
                      <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                        <li>Python 3.11</li>
                        <li>TensorFlow 2.20</li>
                        <li>Keras</li>
                        <li>NumPy, Pandas</li>
                        <li>Scikit-learn</li>
                        <li>OpenCV</li>
                        <li>Matplotlib, Seaborn</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">Frontend / Interface</h4>
                      <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                        <li>React 19</li>
                        <li>TypeScript</li>
                        <li>Tailwind CSS 4</li>
                        <li>tRPC</li>
                        <li>shadcn/ui</li>
                        <li>Wouter (routing)</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">Estrutura do Projeto</h3>
                  <div className="bg-gray-50 p-4 rounded-lg font-mono text-sm">
                    <pre className="text-gray-700">
{`CardioIA/
├── data/
│   ├── raw_images/          # Imagens originais
│   └── processed/           # Dados pré-processados
├── notebooks/
│   ├── Parte1_Preprocessamento_Imagens.py
│   └── Parte2_CNN_Otimizado.py
├── models/
│   ├── cnn_simples.h5
│   ├── vgg16_transfer_learning.h5
│   └── resnet50_transfer_learning.h5
├── reports/                 # Visualizações e gráficos
└── interface/               # Interface web (este projeto)`}
                    </pre>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">Fluxo de Trabalho</h3>
                  <div className="space-y-3">
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-blue-600 font-bold">1</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">Coleta e Preparação</h4>
                        <p className="text-sm text-gray-600">
                          Dataset sintético criado com características de imagens médicas
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-cyan-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-cyan-600 font-bold">2</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">Pré-processamento</h4>
                        <p className="text-sm text-gray-600">
                          Aplicação de técnicas de normalização e equalização
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-purple-600 font-bold">3</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">Treinamento</h4>
                        <p className="text-sm text-gray-600">
                          Treinamento de 3 modelos com diferentes arquiteturas
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-green-600 font-bold">4</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">Avaliação</h4>
                        <p className="text-sm text-gray-600">
                          Análise de métricas e comparação de desempenho
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center flex-shrink-0">
                        <span className="text-orange-600 font-bold">5</span>
                      </div>
                      <div>
                        <h4 className="font-semibold">Interface</h4>
                        <p className="text-sm text-gray-600">
                          Desenvolvimento de interface web para visualização
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-3">Entregáveis</h3>
                  <div className="grid md:grid-cols-2 gap-4">
                    <Card className="bg-blue-50 border-blue-200">
                      <CardHeader>
                        <CardTitle className="text-sm">Notebooks Python</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="text-xs text-gray-600 space-y-1">
                          <li>✓ Parte1_Preprocessamento_Imagens.py</li>
                          <li>✓ Parte2_CNN_Otimizado.py</li>
                        </ul>
                      </CardContent>
                    </Card>
                    <Card className="bg-cyan-50 border-cyan-200">
                      <CardHeader>
                        <CardTitle className="text-sm">Modelos Treinados</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="text-xs text-gray-600 space-y-1">
                          <li>✓ cnn_simples.h5 (295 MB)</li>
                          <li>✓ vgg16_transfer_learning.h5 (57 MB)</li>
                          <li>✓ resnet50_transfer_learning.h5 (94 MB)</li>
                        </ul>
                      </CardContent>
                    </Card>
                    <Card className="bg-purple-50 border-purple-200">
                      <CardHeader>
                        <CardTitle className="text-sm">Visualizações</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="text-xs text-gray-600 space-y-1">
                          <li>✓ Amostras do dataset</li>
                          <li>✓ Comparação de métricas</li>
                          <li>✓ Matrizes de confusão</li>
                          <li>✓ Histórico de treinamento</li>
                        </ul>
                      </CardContent>
                    </Card>
                    <Card className="bg-green-50 border-green-200">
                      <CardHeader>
                        <CardTitle className="text-sm">Interface Web</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <ul className="text-xs text-gray-600 space-y-1">
                          <li>✓ Dashboard interativo</li>
                          <li>✓ Visualização de resultados</li>
                          <li>✓ Documentação completa</li>
                        </ul>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
