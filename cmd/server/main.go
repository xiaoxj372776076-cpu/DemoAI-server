package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/xiaoxj372776076-cpu/DemoAI-server/internal/asr"
	"github.com/xiaoxj372776076-cpu/DemoAI-server/internal/httpapi"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	port := envOrDefault("PORT", "8080")
	origin := envOrDefault("ALLOWED_ORIGIN", "http://localhost:4173")
	dataRoot := envOrDefault("DATA_ROOT", defaultDataRoot())
	dataRepository := absolutePath(envOrDefault("DEMOAI_DATA_REPO", "../DemoAI-data"))
	python := envOrDefault("ASR_PYTHON", filepath.Join(dataRepository, ".venv312", "bin", "python"))
	model := envOrDefault("ASR_MODEL", "mlx-community/whisper-small-mlx")
	modelCache := envOrDefault("ASR_MODEL_CACHE", filepath.Join(dataRoot, "models", "asr"))
	maxUploadBytes := envInt64OrDefault("MAX_UPLOAD_BYTES", 500<<20)

	handler := httpapi.New(httpapi.Config{
		AllowedOrigin:  origin,
		DataRoot:       dataRoot,
		MaxUploadBytes: maxUploadBytes,
		Transcriber: asr.CommandRunner{
			Python:     python,
			Script:     filepath.Join(dataRepository, "operators", "asr", "transcribe_video.py"),
			WorkingDir: dataRepository,
			Model:      model,
			ModelCache: modelCache,
		},
	}, logger)

	server := &http.Server{
		Addr:              ":" + port,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       5 * time.Minute,
		WriteTimeout:      5 * time.Minute,
		IdleTimeout:       60 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		logger.Info("server started",
			"address", server.Addr,
			"allowed_origin", origin,
			"data_root", dataRoot,
			"demoai_data_repo", dataRepository,
		)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server stopped unexpectedly", "error", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownCtx); err != nil {
		logger.Error("graceful shutdown failed", "error", err)
		os.Exit(1)
	}
	logger.Info("server stopped")
}

func defaultDataRoot() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "DemoAI-TrainingData"
	}
	return filepath.Join(home, "Desktop", "DemoAI-TrainingData")
}

func absolutePath(path string) string {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return path
	}
	return absolute
}

func envOrDefault(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func envInt64OrDefault(key string, fallback int64) int64 {
	value := os.Getenv(key)
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseInt(value, 10, 64)
	if err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}
