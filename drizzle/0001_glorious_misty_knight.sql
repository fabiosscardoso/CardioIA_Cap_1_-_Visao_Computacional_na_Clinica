CREATE TABLE `model_metrics` (
	`id` int AUTO_INCREMENT NOT NULL,
	`modelo` varchar(64) NOT NULL,
	`acuracia` float NOT NULL,
	`precisao` float NOT NULL,
	`recall` float NOT NULL,
	`f1Score` float NOT NULL,
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `model_metrics_id` PRIMARY KEY(`id`),
	CONSTRAINT `model_metrics_modelo_unique` UNIQUE(`modelo`)
);
--> statement-breakpoint
CREATE TABLE `predictions` (
	`id` int AUTO_INCREMENT NOT NULL,
	`userId` int NOT NULL,
	`imageUrl` text NOT NULL,
	`modelo` varchar(64) NOT NULL,
	`classePredict` varchar(64) NOT NULL,
	`confianca` float NOT NULL,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `predictions_id` PRIMARY KEY(`id`)
);
