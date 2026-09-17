package httpapi

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	maxRequestBody = 1 << 20
	plusUpgradeURL = "https://chatgpt.com/plans/plus/"
	defaultMaxFile = 500 << 20
)

var allowedMediaExtensions = map[string]bool{
	".aac": true, ".flac": true, ".m4a": true, ".mkv": true,
	".mov": true, ".mp3": true, ".mp4": true, ".ogg": true,
	".wav": true, ".webm": true,
}

type Transcriber interface {
	Transcribe(ctx context.Context, inputPath, outputPath string) error
}

type Config struct {
	AllowedOrigin  string
	DataRoot       string
	MaxUploadBytes int64
	Transcriber    Transcriber
}

type api struct {
	allowedOrigin  string
	dataRoot       string
	maxUploadBytes int64
	transcriber    Transcriber
	jobs           *jobStore
	logger         *slog.Logger
}

type job struct {
	ID             string        `json:"id"`
	Operator       string        `json:"operator"`
	OriginalName   string        `json:"original_name"`
	Status         string        `json:"status"`
	Stage          string        `json:"stage"`
	Progress       int           `json:"progress"`
	CreatedAt      time.Time     `json:"created_at"`
	CompletedAt    *time.Time    `json:"completed_at,omitempty"`
	Error          *apiError     `json:"error,omitempty"`
	Artifacts      *jobArtifacts `json:"artifacts,omitempty"`
	inputPath      string
	transcriptPath string
}

type jobArtifacts struct {
	TranscriptURL string `json:"transcript_url"`
}

type jobStore struct {
	mu   sync.RWMutex
	jobs map[string]job
}

type upgradeRequest struct {
	Account  string `json:"account,omitempty"`
	Email    string `json:"email,omitempty"`
	Password string `json:"password,omitempty"`
}

type response struct {
	Success    bool      `json:"success"`
	Status     string    `json:"status,omitempty"`
	Message    string    `json:"message,omitempty"`
	UpgradeURL string    `json:"upgrade_url,omitempty"`
	Error      *apiError `json:"error,omitempty"`
}

type apiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type productItem struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	URL         string `json:"url,omitempty"`
	Enabled     bool   `json:"enabled"`
}

type productCatalog struct {
	Items []productItem `json:"items"`
}

type operatorPricing struct {
	Currency string  `json:"currency"`
	Amount   float64 `json:"amount"`
	Unit     string  `json:"unit"`
	Display  string  `json:"display"`
	Note     string  `json:"note"`
}

type operatorDefinition struct {
	ID                 string          `json:"id"`
	Name               string          `json:"name"`
	Category           string          `json:"category"`
	Description        string          `json:"description"`
	LongDescription    string          `json:"long_description"`
	URL                string          `json:"url"`
	Available          bool            `json:"available"`
	AcceptedExtensions []string        `json:"accepted_extensions"`
	MaxUploadBytes     int64           `json:"max_upload_bytes"`
	Pricing            operatorPricing `json:"pricing"`
}

type operatorCatalog struct {
	Items []operatorDefinition `json:"items"`
}

func New(config Config, logger *slog.Logger) http.Handler {
	if logger == nil {
		logger = slog.Default()
	}
	if config.MaxUploadBytes <= 0 {
		config.MaxUploadBytes = defaultMaxFile
	}

	app := &api{
		allowedOrigin:  strings.TrimSpace(config.AllowedOrigin),
		dataRoot:       filepath.Clean(config.DataRoot),
		maxUploadBytes: config.MaxUploadBytes,
		transcriber:    config.Transcriber,
		jobs:           &jobStore{jobs: make(map[string]job)},
		logger:         logger,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", app.health)
	mux.HandleFunc("GET /api/v1/catalog/products", app.getProducts)
	mux.HandleFunc("GET /api/v1/operators", app.getOperators)
	mux.HandleFunc("GET /api/v1/operators/{id}", app.getOperator)
	mux.HandleFunc("POST /api/v1/jobs", app.createJob)
	mux.HandleFunc("GET /api/v1/jobs/{id}", app.getJob)
	mux.HandleFunc("GET /api/v1/jobs/{id}/artifacts/transcript", app.getTranscript)
	mux.HandleFunc("POST /api/v1/chatgpt-plus/upgrade", app.createUpgradeSession)
	mux.HandleFunc("OPTIONS /api/v1/{path...}", app.preflight)

	return app.middleware(mux)
}

func (a *api) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"success": true,
		"status":  "ok",
		"service": "demoai-asr",
		"ready":   a.transcriber != nil && a.dataRoot != ".",
	})
}

func (a *api) getProducts(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, productCatalog{Items: []productItem{
		{
			ID:          "data-engine",
			Name:        "数据引擎",
			Description: "采集、治理与交付全链路",
			Enabled:     false,
		},
		{
			ID:          "model-evaluation",
			Name:        "模型评测",
			Description: "从能力到安全的系统评估",
			Enabled:     false,
		},
		{
			ID:          "operator-marketplace",
			Name:        "算子广场",
			Description: "浏览并运行可用的数据处理算子",
			URL:         "/operators.html",
			Enabled:     true,
		},
	}})
}

func (a *api) getOperators(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, operatorCatalog{Items: []operatorDefinition{a.asrOperator()}})
}

func (a *api) getOperator(w http.ResponseWriter, r *http.Request) {
	if r.PathValue("id") != "asr" {
		writeAPIError(w, http.StatusNotFound, "operator_not_found", "The requested operator does not exist.")
		return
	}
	writeJSON(w, http.StatusOK, a.asrOperator())
}

func (a *api) asrOperator() operatorDefinition {
	return operatorDefinition{
		ID:              "asr",
		Name:            "ASR 语音转写",
		Category:        "音视频理解",
		Description:     "把视频或音频中的语音转换为带时间戳的结构化文本。",
		LongDescription: "基于 DemoAI-data 的 Whisper 算子完成语种识别、分段转写与 JSON 结果交付。",
		URL:             "/asr.html",
		Available:       a.transcriber != nil,
		AcceptedExtensions: []string{
			".mp4", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
		},
		MaxUploadBytes: a.maxUploadBytes,
		Pricing: operatorPricing{
			Currency: "CNY",
			Amount:   3,
			Unit:     "media_hour",
			Display:  "¥3.00 / 数据小时",
			Note:     "按上传媒体的实际时长计费，本地演示不会产生真实费用。",
		},
	}
}

func (a *api) createJob(w http.ResponseWriter, r *http.Request) {
	if a.transcriber == nil {
		writeAPIError(w, http.StatusServiceUnavailable, "asr_unavailable", "ASR service is not configured.")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, a.maxUploadBytes+(2<<20))
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		var maxBytesError *http.MaxBytesError
		if errors.As(err, &maxBytesError) {
			writeAPIError(w, http.StatusRequestEntityTooLarge, "file_too_large", "The uploaded file is too large.")
			return
		}
		writeAPIError(w, http.StatusBadRequest, "invalid_multipart", "Upload a media file using multipart form data.")
		return
	}
	if r.MultipartForm != nil {
		defer r.MultipartForm.RemoveAll()
	}

	operator := strings.TrimSpace(r.FormValue("operator"))
	if operator == "" {
		operator = "asr"
	}
	if operator != "asr" {
		writeAPIError(w, http.StatusBadRequest, "unsupported_operator", "Only the asr operator is available.")
		return
	}

	upload, header, err := r.FormFile("file")
	if err != nil {
		writeAPIError(w, http.StatusBadRequest, "file_required", "Select a video or audio file.")
		return
	}
	defer upload.Close()
	if header.Size > a.maxUploadBytes {
		writeAPIError(w, http.StatusRequestEntityTooLarge, "file_too_large", "The uploaded file is too large.")
		return
	}

	extension := strings.ToLower(filepath.Ext(header.Filename))
	if !allowedMediaExtensions[extension] {
		writeAPIError(w, http.StatusBadRequest, "unsupported_media", "Supported formats: MP4, MOV, MKV, WebM, MP3, WAV, M4A, AAC, FLAC, and OGG.")
		return
	}

	jobID, err := newJobID()
	if err != nil {
		writeAPIError(w, http.StatusInternalServerError, "job_id_failed", "Could not create a job.")
		return
	}
	jobDirectory := filepath.Join(a.dataRoot, "jobs", jobID)
	inputDirectory := filepath.Join(jobDirectory, "input")
	outputDirectory := filepath.Join(jobDirectory, "output")
	if err := os.MkdirAll(inputDirectory, 0o755); err != nil {
		writeAPIError(w, http.StatusInternalServerError, "storage_failed", "Could not prepare job storage.")
		return
	}
	if err := os.MkdirAll(outputDirectory, 0o755); err != nil {
		writeAPIError(w, http.StatusInternalServerError, "storage_failed", "Could not prepare job storage.")
		return
	}

	inputPath := filepath.Join(inputDirectory, "source"+extension)
	if err := saveUpload(upload, inputPath, a.maxUploadBytes); err != nil {
		if errors.Is(err, errUploadTooLarge) {
			writeAPIError(w, http.StatusRequestEntityTooLarge, "file_too_large", "The uploaded file is too large.")
			return
		}
		a.logger.Error("failed to save upload", "job_id", jobID, "error", err)
		writeAPIError(w, http.StatusInternalServerError, "storage_failed", "Could not save the uploaded file.")
		return
	}

	transcriptPath := filepath.Join(outputDirectory, "transcript.json")
	created := job{
		ID:             jobID,
		Operator:       "asr",
		OriginalName:   filepath.Base(header.Filename),
		Status:         "queued",
		Stage:          "queued",
		Progress:       10,
		CreatedAt:      time.Now().UTC(),
		inputPath:      inputPath,
		transcriptPath: transcriptPath,
	}
	a.jobs.put(created)
	go a.runJob(jobID)

	w.Header().Set("Location", "/api/v1/jobs/"+jobID)
	writeJSON(w, http.StatusAccepted, created)
}

func (a *api) getJob(w http.ResponseWriter, r *http.Request) {
	item, ok := a.jobs.get(r.PathValue("id"))
	if !ok {
		writeAPIError(w, http.StatusNotFound, "job_not_found", "The requested job does not exist.")
		return
	}
	writeJSON(w, http.StatusOK, item)
}

func (a *api) getTranscript(w http.ResponseWriter, r *http.Request) {
	item, ok := a.jobs.get(r.PathValue("id"))
	if !ok {
		writeAPIError(w, http.StatusNotFound, "job_not_found", "The requested job does not exist.")
		return
	}
	if item.Status != "succeeded" || item.transcriptPath == "" {
		writeAPIError(w, http.StatusConflict, "artifact_not_ready", "The transcript is not ready.")
		return
	}
	w.Header().Set("Content-Disposition", fmt.Sprintf(`inline; filename="%s-transcript.json"`, item.ID))
	http.ServeFile(w, r, item.transcriptPath)
}

func (a *api) runJob(jobID string) {
	item, ok := a.jobs.get(jobID)
	if !ok {
		return
	}
	a.jobs.update(jobID, func(current *job) {
		current.Status = "processing"
		current.Stage = "transcribing"
		current.Progress = 45
	})

	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Minute)
	defer cancel()
	err := a.transcriber.Transcribe(ctx, item.inputPath, item.transcriptPath)
	completedAt := time.Now().UTC()
	if err != nil {
		a.logger.Error("ASR job failed", "job_id", jobID, "error", err)
		a.jobs.update(jobID, func(current *job) {
			current.Status = "failed"
			current.Stage = "failed"
			current.Progress = 100
			current.CompletedAt = &completedAt
			current.Error = &apiError{Code: "asr_failed", Message: "ASR 转写失败，请查看服务端日志。"}
		})
		return
	}

	a.jobs.update(jobID, func(current *job) {
		current.Status = "succeeded"
		current.Stage = "completed"
		current.Progress = 100
		current.CompletedAt = &completedAt
		current.Artifacts = &jobArtifacts{TranscriptURL: "/api/v1/jobs/" + jobID + "/artifacts/transcript"}
	})
}

func (a *api) createUpgradeSession(w http.ResponseWriter, r *http.Request) {
	if mediaType := strings.ToLower(strings.TrimSpace(strings.Split(r.Header.Get("Content-Type"), ";")[0])); mediaType != "application/json" {
		writeAPIError(w, http.StatusUnsupportedMediaType, "unsupported_media_type", "Content-Type must be application/json.")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxRequestBody)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()

	var input upgradeRequest
	if err := decoder.Decode(&input); err != nil {
		writeAPIError(w, http.StatusBadRequest, "invalid_request", decodeErrorMessage(err))
		return
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeAPIError(w, http.StatusBadRequest, "invalid_request", "Request body must contain one JSON object.")
		return
	}

	if input.Account != "" || input.Email != "" || input.Password != "" {
		writeJSON(w, http.StatusBadRequest, response{
			Success:    false,
			Status:     "credentials_rejected",
			UpgradeURL: plusUpgradeURL,
			Error: &apiError{
				Code:    "credentials_not_accepted",
				Message: "Do not send ChatGPT account credentials. Sign in and complete payment directly on OpenAI.",
			},
		})
		return
	}

	writeJSON(w, http.StatusOK, response{
		Success:    true,
		Status:     "user_action_required",
		Message:    "Open the official upgrade page and complete sign-in and payment directly with OpenAI.",
		UpgradeURL: plusUpgradeURL,
	})
}

func (a *api) preflight(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusNoContent)
}

func (a *api) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "no-referrer")
		if a.allowedOrigin != "" {
			w.Header().Set("Access-Control-Allow-Origin", a.allowedOrigin)
		}
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.Header().Add("Vary", "Origin")

		next.ServeHTTP(w, r)
		a.logger.Info("request completed",
			"method", r.Method,
			"path", r.URL.Path,
			"duration_ms", time.Since(started).Milliseconds(),
		)
	})
}

var errUploadTooLarge = errors.New("upload too large")

func saveUpload(source multipart.File, destination string, limit int64) error {
	file, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	defer file.Close()

	written, err := io.Copy(file, io.LimitReader(source, limit+1))
	if err != nil {
		return err
	}
	if written > limit {
		return errUploadTooLarge
	}
	return nil
}

func newJobID() (string, error) {
	random := make([]byte, 8)
	if _, err := rand.Read(random); err != nil {
		return "", err
	}
	return "job_" + hex.EncodeToString(random), nil
}

func (s *jobStore) put(item job) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.jobs[item.ID] = item
}

func (s *jobStore) get(id string) (job, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	item, ok := s.jobs[id]
	return item, ok
}

func (s *jobStore) update(id string, update func(*job)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	item, ok := s.jobs[id]
	if !ok {
		return
	}
	update(&item)
	s.jobs[id] = item
}

func writeAPIError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, response{
		Success: false,
		Status:  "failed",
		Error: &apiError{
			Code:    code,
			Message: message,
		},
	})
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		slog.Error("failed to encode response", "error", err)
	}
}

func decodeErrorMessage(err error) string {
	var maxBytesError *http.MaxBytesError
	if errors.As(err, &maxBytesError) {
		return "Request body is too large."
	}
	return "Request body must be a valid JSON object using supported fields."
}
