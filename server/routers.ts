import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import { getAllModelMetrics, getUserPredictions } from "./db";
import { z } from "zod";

export const appRouter = router({
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),

  // Rotas para métricas dos modelos
  models: router({
    getMetrics: publicProcedure.query(async () => {
      const metrics = await getAllModelMetrics();
      return metrics;
    }),
  }),

  // Rotas para predições
  predictions: router({
    getUserHistory: protectedProcedure.query(async ({ ctx }) => {
      const predictions = await getUserPredictions(ctx.user.id);
      return predictions;
    }),
  }),
});

export type AppRouter = typeof appRouter;
