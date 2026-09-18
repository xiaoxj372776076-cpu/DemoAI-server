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

	operatorASR      = "asr"
	operatorHandPose = "hand_pose"
)

var allowedMediaExtensions = map[string]bool{
	".aac": true, ".flac": true, ".m4a": true, ".mkv": true,
	".mov": true, ".mp3": true, ".mp4": true, ".ogg": true,
	".wav": true, ".webm": true,
}

// The hand pose operator decodes frames, so audio-only containers are out.
var allowedVideoExtensions = map[string]bool{
	".mkv": true, ".mov": true, ".mp4": true, ".webm": true,
}

type Transcriber interface {
	Transcribe(ctx context.Context, inputPath, outputPath string) error
}

// HandPoseEstimator runs the hand pose operator and reports where it put its
// two artifacts. The paths are input-stem derived, so the caller receives
// them rather than guessing.
type HandPoseEstimator interface {
	Estimate(ctx context.Context, inputPath, outputDirectory string) (videoPath, keypointsPath string, err error)
}

type Config struct {
	AllowedOrigin  string
	DataRoot       string
	MaxUploadBytes int64
	Transcriber    Transcriber
	HandPose       HandPoseEstimator
}

type api struct {
	allowedOrigin  string
	dataRoot       string
	maxUploadBytes int64
	transcriber    Transcriber
	handPose       HandPoseEstimator
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
	videoPath      string
	keypointsPath  string
}

type jobArtifacts struct {
	TranscriptURL string `json:"transcript_url,omitempty"`
	VideoURL      string `json:"video_url,omitempty"`
	KeypointsURL  string `json:"keypoints_url,omitempty"`
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

// productItem is a single entry of the navigation product catalog. The web
// front end owns no product copy of its own: it renders whatever the API
// returns, so adding or hiding a product is a backend-only change.
type productItem struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	URL         string `json:"url,omitempty"`
	Enabled     bool   `json:"enabled"`
}

type productCatalog struct {
	Success bool          `json:"success"`
	Items   []productItem `json:"items"`
}

type operatorPricing struct {
	Currency string  `json:"currency"`
	Amount   float64 `json:"amount"`
	Unit     string  `json:"unit"`
	Display  string  `json:"display"`
	Note     string  `json:"note,omitempty"`
}

type operatorDefinition struct {
	ID                 string           `json:"id"`
	Name               string           `json:"name"`
	Category           string           `json:"category"`
	Description        string           `json:"description"`
	LongDescription    string           `json:"long_description,omitempty"`
	URL                string           `json:"url,omitempty"`
	Available          bool             `json:"available"`
	AcceptedExtensions []string         `json:"accepted_extensions,omitempty"`
	MaxUploadBytes     int64            `json:"max_upload_bytes,omitempty"`
	Pricing            *operatorPricing `json:"pricing,omitempty"`
}

type operatorCatalog struct {
	Success bool                 `json:"success"`
	Items   []operatorDefinition `json:"items"`
}

// productCatalogData is the single source of truth for the navigation menu.
func (a *api) productCatalogData() []productItem {
	return []productItem{
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
			ID:          "agent-factory",
			Name:        "Agent 工场",
			Description: "面向复杂任务的环境构建",
			Enabled:     false,
		},
		{
			ID:          "operator-marketplace",
			Name:        "算子广场",
			Description: "浏览并运行可用的数据处理算子",
			URL:         "operators.html",
			Enabled:     true,
		},
	}
}

// operatorCatalogData lists the operators the server can actually run.
func (a *api) operatorCatalogData() []operatorDefinition {
	return []operatorDefinition{a.asrOperator(), a.handPoseOperator()}
}

func (a *api) handPoseOperator() operatorDefinition {
	return operatorDefinition{
		ID:              operatorHandPose,
		Name:            "手部位姿估计",
		Category:        "视觉理解",
		Description:     "跟踪视频中的双手，输出 21 个关节的关键点并合成骨架视频。",
		LongDescription: "基于 DemoAI-data 的 HaWoR 时序算子完成手部检测、MANO 21 关节解码与骨架渲染，交付逐帧关键点 JSON 和合成视频。",
		URL:             "handpose.html",
		Available:       a.handPose != nil,
		AcceptedExtensions: []string{
			".mkv", ".mov", ".mp4", ".webm",
		},
		MaxUploadBytes: a.maxUploadBytes,
		Pricing: &operatorPricing{
			Currency: "CNY",
			Amount:   6,
			Unit:     "video_hour",
			Display:  "¥6.00 / 数据小时",
			Note:     "按上传视频的实际时长计费，本地演示不会产生真实费用。",
		},
	}
}

func (a *api) asrOperator() operatorDefinition {
	return operatorDefinition{
		ID:              operatorASR,
		Name:            "ASR 语音转写",
		Category:        "音视频理解",
		Description:     "把视频或音频中的语音转换为带时间戳的结构化文本。",
		LongDescription: "基于 DemoAI-data 的 Whisper 算子完成语种识别、分段转写与 JSON 结果交付。",
		URL:             "index.html#playground",
		Available:       a.transcriber != nil,
		AcceptedExtensions: []string{
			".aac", ".flac", ".m4a", ".mkv", ".mov",
			".mp3", ".mp4", ".ogg", ".wav", ".webm",
		},
		MaxUploadBytes: a.maxUploadBytes,
		Pricing: &operatorPricing{
			Currency: "CNY",
			Amount:   3,
			Unit:     "media_hour",
			Display:  "¥3.00 / 数据小时",
			Note:     "按上传媒体的实际时长计费，本地演示不会产生真实费用。",
		},
	}
}

func (a *api) operatorByID(id string) (operatorDefinition, bool) {
	for _, item := range a.operatorCatalogData() {
		if item.ID == id {
			return item, true
		}
	}
	return operatorDefinition{}, false
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
		handPose:       config.HandPose,
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
	mux.HandleFunc("GET /api/v1/jobs/{id}/artifacts/video", app.getVideo)
	mux.HandleFunc("GET /api/v1/jobs/{id}/artifacts/keypoints", app.getKeypoints)
	mux.HandleFunc("POST /api/v1/chatgpt-plus/upgrade", app.createUpgradeSession)
	mux.HandleFunc("OPTIONS /api/v1/{path...}", app.preflight)

	return app.middleware(mux)
}

func (a *api) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"success": true,
		"status":  "ok",
		"service": "demoai-operators",
		"ready":   a.dataRoot != "." && (a.transcriber != nil || a.handPose != nil),
		"operators": map[string]bool{
			operatorASR:      a.transcriber != nil,
			operatorHandPose: a.handPose != nil,
		},
	})
}

func (a *api) getProducts(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, productCatalog{Success: true, Items: a.productCatalogData()})
}

func (a *api) getOperators(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, operatorCatalog{Success: true, Items: a.operatorCatalogData()})
}

func (a *api) getOperator(w http.ResponseWriter, r *http.Request) {
	item, ok := a.operatorByID(r.PathValue("id"))
	if !ok {
		writeAPIError(w, http.StatusNotFound, "operator_not_found", "The requested operator does not exist.")
		return
	}
	writeJSON(w, http.StatusOK, item)
}

func (a *api) createJob(w http.ResponseWriter, r *http.Request) {
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
		operator = operatorASR
	}
	definition, ok := a.operatorByID(operator)
	if !ok {
		writeAPIError(w, http.StatusBadRequest, "unsupported_operator", "The requested operator does not exist.")
		return
	}
	if !definition.Available {
		writeAPIError(w, http.StatusServiceUnavailable, definition.ID+"_unavailable", "This operator is not configured on the server.")
		return
	}
	accepted := allowedMediaExtensions
	if operator == operatorHandPose {
		accepted = allowedVideoExtensions
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
	if !accepted[extension] {
		writeAPIError(w, http.StatusBadRequest, "unsupported_media", "Supported formats: "+strings.Join(definition.AcceptedExtensions, ", ")+".")
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

	created := job{
		ID:           jobID,
		Operator:     operator,
		OriginalName: filepath.Base(header.Filename),
		Status:       "queued",
		Stage:        "queued",
		Progress:     10,
		CreatedAt:    time.Now().UTC(),
		inputPath:    inputPath,
	}
	switch operator {
	case operatorHandPose:
		created.videoPath = filepath.Join(outputDirectory, "source_hand_pose.mp4")
		created.keypointsPath = filepath.Join(outputDirectory, "source_hand_keypoints.json")
	default:
		created.transcriptPath = filepath.Join(outputDirectory, "transcript.json")
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

	var err error
	switch item.Operator {
	case operatorHandPose:
		err = a.runHandPoseJob(ctx, jobID, item)
	default:
		err = a.transcriber.Transcribe(ctx, item.inputPath, item.transcriptPath)
	}
	if err != nil {
		a.logger.Error("operator job failed", "job_id", jobID, "operator", item.Operator, "error", err)
		completedAt := time.Now().UTC()
		a.jobs.update(jobID, func(current *job) {
			current.Status = "failed"
			current.Stage = "failed"
			current.Progress = 100
			current.CompletedAt = &completedAt
			current.Error = &apiError{
				Code:    item.Operator + "_failed",
				Message: "算子执行失败，请查看服务端日志。",
			}
		})
		return
	}

	completedAt := time.Now().UTC()
	a.jobs.update(jobID, func(current *job) {
		current.Status = "succeeded"
		current.Stage = "completed"
		current.Progress = 100
		current.CompletedAt = &completedAt
		current.Artifacts = artifactsFor(jobID, current.Operator)
	})
}

func (a *api) runHandPoseJob(ctx context.Context, jobID string, item job) error {
	a.jobs.update(jobID, func(current *job) {
		current.Stage = "estimating_hands"
		current.Progress = 30
	})
	producedVideo, producedKeypoints, err := a.handPose.Estimate(ctx, item.inputPath, filepath.Dir(item.videoPath))
	if err != nil {
		return err
	}
	a.jobs.update(jobID, func(current *job) {
		current.videoPath = producedVideo
		current.keypointsPath = producedKeypoints
	})
	return nil
}

func artifactsFor(jobID, operator string) *jobArtifacts {
	switch operator {
	case operatorHandPose:
		return &jobArtifacts{
			VideoURL:     "/api/v1/jobs/" + jobID + "/artifacts/video",
			KeypointsURL: "/api/v1/jobs/" + jobID + "/artifacts/keypoints",
		}
	default:
		return &jobArtifacts{TranscriptURL: "/api/v1/jobs/" + jobID + "/artifacts/transcript"}
	}
}

func (a *api) getVideo(w http.ResponseWriter, r *http.Request) {
	item, ok := a.jobs.get(r.PathValue("id"))
	if !ok {
		writeAPIError(w, http.StatusNotFound, "job_not_found", "The requested job does not exist.")
		return
	}
	if item.Status != "succeeded" || item.videoPath == "" {
		writeAPIError(w, http.StatusConflict, "artifact_not_ready", "The rendered video is not ready.")
		return
	}
	// ServeContent would otherwise keep the JSON content type set by the
	// middleware, which breaks inline playback in the browser.
	w.Header().Set("Content-Type", "video/mp4")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`inline; filename="%s-hand-pose.mp4"`, item.ID))
	http.ServeFile(w, r, item.videoPath)
}

func (a *api) getKeypoints(w http.ResponseWriter, r *http.Request) {
	item, ok := a.jobs.get(r.PathValue("id"))
	if !ok {
		writeAPIError(w, http.StatusNotFound, "job_not_found", "The requested job does not exist.")
		return
	}
	if item.Status != "succeeded" || item.keypointsPath == "" {
		writeAPIError(w, http.StatusConflict, "artifact_not_ready", "The keypoints are not ready.")
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Content-Disposition", fmt.Sprintf(`inline; filename="%s-hand-keypoints.json"`, item.ID))
	http.ServeFile(w, r, item.keypointsPath)
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
