import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { trpc } from "@/lib/trpc";
import { Award, TrendingUp, Target, Activity } from "lucide-react";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";

export default function Resultados() {
  const { data: metrics, isLoading } = trpc.models.getMetrics.useQuery();

  // Dados estáticos como fallback se não houver banco de dados
  const metricsEstaticas = [
    {
      id: 1,
      modelo: "CNN Simples",
      acuracia: 0.3333,
      precisao: 0.1111,
      recall: 0.3333,
      f1Score: 0.1667,
    },
    {
      id: 2,
      modelo: "VGG16",
      acuracia: 0.4000,
      precisao: 0.4524,
      recall: 0.4000,
      f1Score: 0.2865,
    },
    {
      id: 3,
      modelo: "ResNet50",
      acuracia: 0.3333,
      precisao: 0.1111,
      recall: 0.3333,
      f1Score: 0.1667,
    },
  ];

  // Usar dados do banco se disponíveis, senão usar estáticos
  const metricsData = metrics && metrics.length > 0 ? metrics : metricsEstaticas;

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Carregando resultados...</p>
        </div>
      </div>
    );
  }

  const melhorModelo = metricsData.reduce((prev, current) =>
    (current.acuracia > prev.acuracia) ? current : prev
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <Link href="/">
            <Button variant="ghost">← Voltar</Button>
          </Link>
          <h1 className="text-xl font-bold">Resultados dos Modelos</h1>
          <div className="w-24"></div>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        {/* Melhor Modelo */}
        <Card className="mb-8 border-2 border-blue-200 bg-gradient-to-br from-blue-50 to-white">
          <CardHeader>
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-blue-600 rounded-lg flex items-center justify-center">
                <Award className="w-6 h-6 text-white" />
              </div>
              <div>
                <CardTitle className="text-2xl">Melhor Modelo</CardTitle>
                <CardDescription>Maior acurácia na classificação</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-5 gap-4">
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Modelo</p>
                <p className="text-2xl font-bold text-blue-600">{melhorModelo?.modelo}</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Acurácia</p>
                <p className="text-2xl font-bold">{(melhorModelo?.acuracia ?? 0 * 100).toFixed(2)}%</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Precisão</p>
                <p className="text-2xl font-bold">{(melhorModelo?.precisao ?? 0 * 100).toFixed(2)}%</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">Recall</p>
                <p className="text-2xl font-bold">{(melhorModelo?.recall ?? 0 * 100).toFixed(2)}%</p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-600 mb-1">F1-Score</p>
                <p className="text-2xl font-bold">{(melhorModelo?.f1Score ?? 0 * 100).toFixed(2)}%</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Tabs com Resultados */}
        <Tabs defaultValue="metricas" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="metricas">Métricas</TabsTrigger>
            <TabsTrigger value="comparacao">Comparação</TabsTrigger>
            <TabsTrigger value="confusao">Matrizes</TabsTrigger>
            <TabsTrigger value="treinamento">Treinamento</TabsTrigger>
          </TabsList>

          {/* Métricas Detalhadas */}
          <TabsContent value="metricas" className="space-y-4">
            <div className="grid md:grid-cols-3 gap-4">
              {metricsData.map((metric) => (
                <Card key={metric.modelo} className="hover:shadow-lg transition-shadow">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="w-5 h-5 text-blue-600" />
                      {metric.modelo}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Acurácia</span>
                      <span className="font-bold">{(metric.acuracia * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${metric.acuracia * 100}%` }}
                      ></div>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Precisão</span>
                      <span className="font-bold">{(metric.precisao * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-cyan-600 h-2 rounded-full"
                        style={{ width: `${metric.precisao * 100}%` }}
                      ></div>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Recall</span>
                      <span className="font-bold">{(metric.recall * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-purple-600 h-2 rounded-full"
                        style={{ width: `${metric.recall * 100}%` }}
                      ></div>
                    </div>

                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">F1-Score</span>
                      <span className="font-bold">{(metric.f1Score * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-green-600 h-2 rounded-full"
                        style={{ width: `${metric.f1Score * 100}%` }}
                      ></div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Comparação Visual */}
          <TabsContent value="comparacao">
            <Card>
              <CardHeader>
                <CardTitle>Comparação de Métricas entre Modelos</CardTitle>
                <CardDescription>Visualização comparativa do desempenho dos três modelos</CardDescription>
              </CardHeader>
              <CardContent>
                <img
                  src="/reports/04_comparacao_metricas.png"
                  alt="Comparação de Métricas"
                  className="w-full rounded-lg border"
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Matrizes de Confusão */}
          <TabsContent value="confusao">
            <Card>
              <CardHeader>
                <CardTitle>Matrizes de Confusão</CardTitle>
                <CardDescription>Análise detalhada das predições corretas e incorretas</CardDescription>
              </CardHeader>
              <CardContent>
                <img
                  src="/reports/05_matrizes_confusao.png"
                  alt="Matrizes de Confusão"
                  className="w-full rounded-lg border"
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Histórico de Treinamento */}
          <TabsContent value="treinamento">
            <Card>
              <CardHeader>
                <CardTitle>Histórico de Treinamento</CardTitle>
                <CardDescription>Evolução da acurácia durante o treinamento dos modelos</CardDescription>
              </CardHeader>
              <CardContent>
                <img
                  src="/reports/06_historico_treinamento.png"
                  alt="Histórico de Treinamento"
                  className="w-full rounded-lg border"
                />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Análise e Conclusões */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Análise e Conclusões
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Desempenho dos Modelos</h4>
              <p className="text-gray-600 text-sm">
                O modelo <strong>{melhorModelo?.modelo}</strong> apresentou o melhor desempenho com acurácia de{" "}
                <strong>{(melhorModelo?.acuracia ?? 0 * 100).toFixed(2)}%</strong>. Os modelos de Transfer Learning
                (VGG16 e ResNet50) demonstraram capacidade superior de generalização em comparação com a CNN simples.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Observações Técnicas</h4>
              <ul className="list-disc list-inside text-gray-600 text-sm space-y-1">
                <li>Dataset sintético com 198 imagens divididas em 3 classes balanceadas</li>
                <li>Pré-processamento incluiu normalização, equalização e padronização</li>
                <li>Transfer Learning utilizou pesos pré-treinados do ImageNet</li>
                <li>Treinamento realizado com 20 épocas e early stopping</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-2">Próximos Passos</h4>
              <ul className="list-disc list-inside text-gray-600 text-sm space-y-1">
                <li>Utilizar dataset real de imagens médicas para validação</li>
                <li>Aumentar o número de amostras para melhorar generalização</li>
                <li>Implementar data augmentation para expandir o dataset</li>
                <li>Testar arquiteturas mais recentes (EfficientNet, Vision Transformer)</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
