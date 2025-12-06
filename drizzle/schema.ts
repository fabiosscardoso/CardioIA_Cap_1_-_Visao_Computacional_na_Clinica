import { int, mysqlEnum, mysqlTable, text, timestamp, varchar, float } from "drizzle-orm/mysql-core";

export const users = mysqlTable("users", {
  id: int("id").autoincrement().primaryKey(),
  openId: varchar("openId", { length: 64 }).notNull().unique(),
  name: text("name"),
  email: varchar("email", { length: 320 }),
  loginMethod: varchar("loginMethod", { length: 64 }),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * Tabela para armazenar histórico de predições
 */
export const predictions = mysqlTable("predictions", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  imageUrl: text("imageUrl").notNull(),
  modelo: varchar("modelo", { length: 64 }).notNull(), // CNN Simples, VGG16, ResNet50
  classePredict: varchar("classePredict", { length: 64 }).notNull(), // Normal, Cardiomegalia, Outras_Patologias
  confianca: float("confianca").notNull(), // 0.0 - 1.0
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type Prediction = typeof predictions.$inferSelect;
export type InsertPrediction = typeof predictions.$inferInsert;

/**
 * Tabela para armazenar métricas dos modelos
 */
export const modelMetrics = mysqlTable("model_metrics", {
  id: int("id").autoincrement().primaryKey(),
  modelo: varchar("modelo", { length: 64 }).notNull().unique(),
  acuracia: float("acuracia").notNull(),
  precisao: float("precisao").notNull(),
  recall: float("recall").notNull(),
  f1Score: float("f1Score").notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type ModelMetric = typeof modelMetrics.$inferSelect;
export type InsertModelMetric = typeof modelMetrics.$inferInsert;
