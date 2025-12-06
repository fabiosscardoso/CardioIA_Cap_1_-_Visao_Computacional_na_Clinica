import { useAuth } from "@/_core/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { getLoginUrl } from "@/const";
import { Activity, Brain, BarChart3, FileText, Heart } from "lucide-react";
import { Link } from "wouter";

export default function Home() {
  const { user, isAuthenticated } = useAuth();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-cyan-600 rounded-lg flex items-center justify-center">
              <Heart className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">CardioIA</h1>
              <p className="text-xs text-gray-600">A Nova Era da Cardiologia Inteligente</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {isAuthenticated ? (
              <>
                <span className="text-sm text-gray-600">Olá, {user?.name}</span>
                <Link href="/dashboard">
                  <Button variant="default">Dashboard</Button>
                </Link>
              </>
            ) : (
              <Button asChild>
                <a href={getLoginUrl()}>Entrar</a>
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto text-center space-y-6">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
            <Brain className="w-4 h-4" />
            Projeto Acadêmico - Classificação de Imagens Médicas
          </div>
          <h2 className="text-5xl font-bold text-gray-900 leading-tight">
            Classificação Inteligente de <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-cyan-600">Imagens Cardíacas</span>
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Sistema de classificação de imagens médicas utilizando Redes Neurais Convolucionais (CNN) e Transfer Learning para diagnóstico de patologias cardíacas.
          </p>
          <div className="flex gap-4 justify-center pt-4">
            <Link href="/resultados">
              <Button size="lg" className="gap-2">
                <BarChart3 className="w-5 h-5" />
                Ver Resultados
              </Button>
            </Link>
            <Link href="/documentacao">
              <Button size="lg" variant="outline" className="gap-2">
                <FileText className="w-5 h-5" />
                Documentação
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
          <Card className="border-2 hover:border-blue-200 transition-colors">
            <CardHeader>
              <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-3">
                <Activity className="w-6 h-6 text-blue-600" />
              </div>
              <CardTitle>Pré-processamento</CardTitle>
              <CardDescription>
                Pipeline completo de preparação de imagens médicas com técnicas de normalização, redimensionamento e equalização de histograma.
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="border-2 hover:border-cyan-200 transition-colors">
            <CardHeader>
              <div className="w-12 h-12 bg-cyan-100 rounded-lg flex items-center justify-center mb-3">
                <Brain className="w-6 h-6 text-cyan-600" />
              </div>
              <CardTitle>Modelos CNN</CardTitle>
              <CardDescription>
                Três abordagens de classificação: CNN Simples, VGG16 e ResNet50 com Transfer Learning para comparação de desempenho.
              </CardDescription>
            </CardHeader>
          </Card>

          <Card className="border-2 hover:border-purple-200 transition-colors">
            <CardHeader>
              <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-3">
                <BarChart3 className="w-6 h-6 text-purple-600" />
              </div>
              <CardTitle>Avaliação</CardTitle>
              <CardDescription>
                Métricas detalhadas incluindo acurácia, precisão, recall, F1-score e matrizes de confusão para análise completa.
              </CardDescription>
            </CardHeader>
          </Card>
        </div>
      </section>

      {/* Classes Section */}
      <section className="container mx-auto px-4 py-16 bg-white/50 rounded-2xl my-8">
        <div className="max-w-4xl mx-auto">
          <h3 className="text-3xl font-bold text-center mb-8">Classes de Classificação</h3>
          <div className="grid md:grid-cols-3 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-green-600">Normal</CardTitle>
                <CardDescription>
                  Imagens cardíacas sem evidências de patologias significativas.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-red-600">Cardiomegalia</CardTitle>
                <CardDescription>
                  Aumento anormal do tamanho do coração detectado nas imagens.
                </CardDescription>
              </CardHeader>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-orange-600">Outras Patologias</CardTitle>
                <CardDescription>
                  Outras condições cardíacas identificadas através da análise.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="container mx-auto px-4 py-16">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <Card className="text-center">
              <CardHeader>
                <CardTitle className="text-4xl font-bold text-blue-600">198</CardTitle>
                <CardDescription>Imagens Processadas</CardDescription>
              </CardHeader>
            </Card>
            <Card className="text-center">
              <CardHeader>
                <CardTitle className="text-4xl font-bold text-cyan-600">3</CardTitle>
                <CardDescription>Modelos Treinados</CardDescription>
              </CardHeader>
            </Card>
            <Card className="text-center">
              <CardHeader>
                <CardTitle className="text-4xl font-bold text-purple-600">3</CardTitle>
                <CardDescription>Classes</CardDescription>
              </CardHeader>
            </Card>
            <Card className="text-center">
              <CardHeader>
                <CardTitle className="text-4xl font-bold text-green-600">40%</CardTitle>
                <CardDescription>Melhor Acurácia</CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t bg-white/80 backdrop-blur-sm mt-16">
        <div className="container mx-auto px-4 py-8 text-center text-gray-600">
          <p className="text-sm">
            CardioIA - Projeto Acadêmico de Classificação de Imagens Médicas com CNN
          </p>
          <p className="text-xs mt-2">
            Desenvolvido como parte do projeto de Inteligência Artificial aplicada à Cardiologia
          </p>
        </div>
      </footer>
    </div>
  );
}
