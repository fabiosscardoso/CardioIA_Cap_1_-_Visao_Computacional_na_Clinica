# 🚀 Guia de Instalação Local - CardioIA Interface

Este guia explica como baixar e rodar o projeto CardioIA localmente em seu computador.

---

## 📋 Pré-requisitos

Antes de começar, certifique-se de ter instalado:

1. **Node.js** (versão 18 ou superior)
   - Download: https://nodejs.org/
   - Verifique: `node --version`

2. **pnpm** (gerenciador de pacotes)
   - Instalação: `npm install -g pnpm`
   - Verifique: `pnpm --version`

---

## 📦 Instalação

### Passo 1: Instalar Dependências

Dentro da pasta do projeto, execute:

```bash
pnpm install
```

### Passo 2: Configurar Variáveis de Ambiente

**Demonstração:**

Crie um arquivo `.env` na raiz do projeto com:

```env
# Configurações recomendadas para demonstração
VITE_APP_TITLE=CardioIA
VITE_APP_LOGO=❤️
NODE_ENV=development
JWT_SECRET=local-development-secret
OAUTH_SERVER_URL=http://localhost:3000
VITE_OAUTH_PORTAL_URL=http://localhost:3000
VITE_APP_ID=local-dev
OWNER_OPEN_ID=local-owner
OWNER_NAME=Dev Local
VITE_ANALYTICS_ENDPOINT=
VITE_ANALYTICS_WEBSITE_ID=
BUILT_IN_FORGE_API_URL=http://localhost:3000/api
BUILT_IN_FORGE_API_KEY=local-key
VITE_FRONTEND_FORGE_API_KEY=local-key
VITE_FRONTEND_FORGE_API_URL=http://localhost:3000/api
```

## ▶️ Executar o Projeto

### Modo Desenvolvimento

```bash
pnpm dev
```

URL's:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:3000/api

Você verá no terminal:

```
Server running on http://localhost:3000/
[vite] ready in XXX ms
```

### Modo Produção

Para compilar e executar em modo produção:

```bash
# Compilar
pnpm build

# Executar
pnpm start
```

---

## 🌐 Acessar a Interface

```
http://localhost:3000
```

Paginas:
- **Página Inicial:** Apresentação do projeto
- **Ver Resultados:** Dashboard com métricas e gráficos
- **Documentação:** Explicação completa das PARTES 1 e 2

---

## 📁 Estrutura do Projeto

```
cardioia_interface/
├── client/                    # Frontend React
│   ├── src/
│   │   ├── pages/            # Páginas da aplicação
│   │   │   ├── Home.tsx      # Página inicial
│   │   │   ├── Resultados.tsx    # Dashboard de resultados
│   │   │   └── Documentacao.tsx  # Documentação
│   │   ├── components/       # Componentes reutilizáveis
│   │   └── lib/              # Configurações
│   └── public/
│       └── reports/          # Gráficos e visualizações
├── server/                   # Backend Node.js
│   ├── routers.ts           # Rotas da API
│   ├── db.ts                # Funções do banco de dados
│   └── seed.ts              # Script de seed
├── drizzle/                 # Schema do banco de dados
│   └── schema.ts
├── package.json             # Dependências
└── README.md               # Documentação do template
```

---

## 📊 Dados e Modelos

### Visualizações

Os gráficos estão em `client/public/reports/`:
- `01_amostras_dataset.png`
- `02_antes_depois_preprocessamento.png`
- `03_distribuicao_conjuntos.png`
- `04_comparacao_metricas.png`
- `05_matrizes_confusao.png`
- `06_historico_treinamento.png`

---

**Desenvolvido como parte do Projeto CardioIA**

**A Nova Era da Cardiologia Inteligente** ❤️🤖
