import { drizzle } from "drizzle-orm/mysql2";
import { modelMetrics } from "../drizzle/schema.js";

const db = drizzle(process.env.DATABASE_URL!);

const metrics = [
  {
    modelo: "CNN Simples",
    acuracia: 0.3333,
    precisao: 0.1111,
    recall: 0.3333,
    f1Score: 0.1667,
  },
  {
    modelo: "VGG16",
    acuracia: 0.4000,
    precisao: 0.4524,
    recall: 0.4000,
    f1Score: 0.2865,
  },
  {
    modelo: "ResNet50",
    acuracia: 0.3333,
    precisao: 0.1111,
    recall: 0.3333,
    f1Score: 0.1667,
  },
];

async function seed() {
  console.log("Inserindo métricas dos modelos...");
  for (const metric of metrics) {
    await db.insert(modelMetrics).values(metric).onDuplicateKeyUpdate({
      set: metric,
    });
  }
  console.log("✓ Métricas dos modelos inseridas com sucesso!");
  process.exit(0);
}

seed().catch((error) => {
  console.error("Erro ao inserir métricas:", error);
  process.exit(1);
});
