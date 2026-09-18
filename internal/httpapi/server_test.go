package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type fakeTranscriber struct {
	err error
}

func (f fakeTranscriber) Transcribe(_ context.Context, inputPath, outputPath string) error {
	if f.err != nil {
		return f.err
	}
	if _, err := os.Stat(inputPath); err != nil {
		return err
	}
	return os.WriteFile(outputPath, []byte(`{"language":"zh","text":"测试转写","segments":[]}`), 0o644)
}

type fakeHandPose struct {
	err error
}

func (f fakeHandPose) Estimate(_ context.Context, inputPath, outputDirectory string) (string, string, error) {
	if f.err != nil {
		return "", "", f.err
	}
	if _, err := os.Stat(inputPath); err != nil {
		return "", "", err
	}
	stem := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))
	videoPath := filepath.Join(outputDirectory, stem+"_hand_pose.mp4")
	keypointsPath := filepath.Join(outputDirectory, stem+"_hand_keypoints.json")
	if err := os.WriteFile(videoPath, []byte("fake mp4"), 0o644); err != nil {
		return "", "", err
	}
	if err := os.WriteFile(keypointsPath, []byte(`{"metadata":{"frame_count":1},"frames":[]}`), 0o644); err != nil {
		return "", "", err
	}
	return videoPath, keypointsPath, nil
}

func TestHealth(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})
	request := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	var payload map[string]any
	decodeJSON(t, recorder, &payload)
	if payload["status"] != "ok" || payload["ready"] != true {
		t.Fatalf("payload = %+v", payload)
	}
}

func TestProductCatalogIsRenderedFromBackend(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})
	request := httptest.NewRequest(http.MethodGet, "/api/v1/catalog/products", nil)
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	var payload productCatalog
	decodeJSON(t, recorder, &payload)
	if !payload.Success {
		t.Fatalf("success = false, want true")
	}
	if len(payload.Items) != 4 {
		t.Fatalf("items = %+v", payload.Items)
	}

	var marketplace *productItem
	for i := range payload.Items {
		if payload.Items[i].ID == "operator-marketplace" {
			marketplace = &payload.Items[i]
		}
	}
	if marketplace == nil {
		t.Fatal("product catalog is missing the operator marketplace entry")
	}
	if marketplace.Name != "算子广场" || marketplace.URL != "operators.html" || !marketplace.Enabled {
		t.Fatalf("marketplace = %+v", marketplace)
	}
	for _, item := range payload.Items {
		if item.Enabled && item.URL == "" {
			t.Fatalf("enabled product %q has no URL", item.ID)
		}
	}
}

func TestOperatorCatalogReturnsASRWithBackendRules(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})
	request := httptest.NewRequest(http.MethodGet, "/api/v1/operators", nil)
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	var payload operatorCatalog
	decodeJSON(t, recorder, &payload)
	if len(payload.Items) != 2 {
		t.Fatalf("items = %+v", payload.Items)
	}
	asr := payload.Items[0]
	if asr.ID != "asr" || asr.URL != "index.html#playground" {
		t.Fatalf("asr = %+v", asr)
	}
	if !asr.Available {
		t.Fatal("asr operator should be available when a transcriber is configured")
	}
	if asr.MaxUploadBytes != 1<<20 {
		t.Fatalf("max_upload_bytes = %d, want %d", asr.MaxUploadBytes, 1<<20)
	}
	if asr.Pricing == nil || asr.Pricing.Amount != 3 || asr.Pricing.Currency != "CNY" {
		t.Fatalf("pricing = %+v", asr.Pricing)
	}
	if len(asr.AcceptedExtensions) != len(allowedMediaExtensions) {
		t.Fatalf("accepted_extensions = %+v", asr.AcceptedExtensions)
	}

	handPose := payload.Items[1]
	if handPose.ID != "hand_pose" || handPose.URL != "handpose.html" {
		t.Fatalf("hand pose = %+v", handPose)
	}
	if handPose.Pricing == nil || handPose.Pricing.Currency != "CNY" {
		t.Fatalf("hand pose pricing = %+v", handPose.Pricing)
	}
	// Audio-only containers make no sense for a frame decoder, so the hand
	// pose operator must publish a narrower extension list.
	if len(handPose.AcceptedExtensions) != len(allowedVideoExtensions) {
		t.Fatalf("hand pose accepted_extensions = %+v", handPose.AcceptedExtensions)
	}
	for _, extension := range handPose.AcceptedExtensions {
		if !allowedVideoExtensions[extension] {
			t.Fatalf("hand pose advertises non-video extension %q", extension)
		}
	}
}

func TestHandPoseAvailabilityFollowsConfiguration(t *testing.T) {
	without := newTestServer(t, fakeTranscriber{})
	recorder := httptest.NewRecorder()
	without.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/v1/operators/hand_pose", nil))
	var detail operatorDefinition
	decodeJSON(t, recorder, &detail)
	if detail.Available {
		t.Fatal("hand pose should be unavailable when no estimator is configured")
	}

	with := newTestServerWithHandPose(t, fakeTranscriber{}, fakeHandPose{})
	recorder = httptest.NewRecorder()
	with.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/v1/operators/hand_pose", nil))
	decodeJSON(t, recorder, &detail)
	if !detail.Available {
		t.Fatal("hand pose should be available when an estimator is configured")
	}
}

func TestOperatorDetail(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})

	recorder := httptest.NewRecorder()
	server.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/v1/operators/asr", nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	var detail operatorDefinition
	decodeJSON(t, recorder, &detail)
	if detail.ID != "asr" {
		t.Fatalf("detail = %+v", detail)
	}

	missing := httptest.NewRecorder()
	server.ServeHTTP(missing, httptest.NewRequest(http.MethodGet, "/api/v1/operators/unknown", nil))
	if missing.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d", missing.Code, http.StatusNotFound)
	}
	if payload := decodeResponse(t, missing); payload.Error == nil || payload.Error.Code != "operator_not_found" {
		t.Fatalf("payload = %+v", payload)
	}
}

func TestCreateASRJobAndReadArtifact(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})
	request := uploadRequest(t, "sample.mp4", []byte("fake media"), "asr")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want %d: %s", recorder.Code, http.StatusAccepted, recorder.Body.String())
	}
	var created job
	decodeJSON(t, recorder, &created)
	if created.ID == "" || created.Status == "" {
		t.Fatalf("created job = %+v", created)
	}

	completed := waitForJob(t, server, created.ID)
	if completed.Status != "succeeded" || completed.Artifacts == nil {
		t.Fatalf("completed job = %+v", completed)
	}

	artifactRequest := httptest.NewRequest(http.MethodGet, completed.Artifacts.TranscriptURL, nil)
	artifactRecorder := httptest.NewRecorder()
	server.ServeHTTP(artifactRecorder, artifactRequest)
	if artifactRecorder.Code != http.StatusOK {
		t.Fatalf("artifact status = %d", artifactRecorder.Code)
	}
	var transcript map[string]any
	decodeJSON(t, artifactRecorder, &transcript)
	if transcript["text"] != "测试转写" {
		t.Fatalf("transcript = %+v", transcript)
	}
}

func TestCreateJobRejectsUnsupportedOperator(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})
	request := uploadRequest(t, "sample.mp4", []byte("fake media"), "not_an_operator")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
	payload := decodeResponse(t, recorder)
	if payload.Error == nil || payload.Error.Code != "unsupported_operator" {
		t.Fatalf("payload = %+v", payload)
	}
}

func TestCreateHandPoseJobRejectsUnknownOperator(t *testing.T) {
	server := newTestServerWithHandPose(t, fakeTranscriber{}, fakeHandPose{})
	request := uploadRequest(t, "sample.mp4", []byte("fake media"), "not_an_operator")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
}

func TestCreateHandPoseJobWithoutEstimatorIsUnavailable(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})
	request := uploadRequest(t, "sample.mp4", []byte("fake media"), "hand_pose")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusServiceUnavailable)
	}
	payload := decodeResponse(t, recorder)
	if payload.Error == nil || payload.Error.Code != "hand_pose_unavailable" {
		t.Fatalf("payload = %+v", payload)
	}
}

func TestCreateHandPoseJobRejectsAudioOnlyUpload(t *testing.T) {
	server := newTestServerWithHandPose(t, fakeTranscriber{}, fakeHandPose{})
	request := uploadRequest(t, "voice.mp3", []byte("fake audio"), "hand_pose")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
	payload := decodeResponse(t, recorder)
	if payload.Error == nil || payload.Error.Code != "unsupported_media" {
		t.Fatalf("payload = %+v", payload)
	}
}

func TestCreateHandPoseJobAndReadArtifacts(t *testing.T) {
	server := newTestServerWithHandPose(t, fakeTranscriber{}, fakeHandPose{})
	request := uploadRequest(t, "hands.mp4", []byte("fake media"), "hand_pose")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusAccepted {
		t.Fatalf("status = %d, want %d: %s", recorder.Code, http.StatusAccepted, recorder.Body.String())
	}
	var created job
	decodeJSON(t, recorder, &created)
	if created.Operator != "hand_pose" {
		t.Fatalf("operator = %q, want hand_pose", created.Operator)
	}

	completed := waitForJob(t, server, created.ID)
	if completed.Status != "succeeded" || completed.Artifacts == nil {
		t.Fatalf("completed job = %+v", completed)
	}
	if completed.Artifacts.VideoURL == "" || completed.Artifacts.KeypointsURL == "" {
		t.Fatalf("artifacts = %+v", completed.Artifacts)
	}
	if completed.Artifacts.TranscriptURL != "" {
		t.Fatalf("hand pose job should not expose a transcript URL: %+v", completed.Artifacts)
	}

	videoRecorder := httptest.NewRecorder()
	server.ServeHTTP(videoRecorder, httptest.NewRequest(http.MethodGet, completed.Artifacts.VideoURL, nil))
	if videoRecorder.Code != http.StatusOK {
		t.Fatalf("video status = %d", videoRecorder.Code)
	}
	if contentType := videoRecorder.Header().Get("Content-Type"); contentType != "video/mp4" {
		t.Fatalf("content type = %q, want video/mp4", contentType)
	}
	if videoRecorder.Body.String() != "fake mp4" {
		t.Fatalf("video body = %q", videoRecorder.Body.String())
	}

	keypointsRecorder := httptest.NewRecorder()
	server.ServeHTTP(keypointsRecorder, httptest.NewRequest(http.MethodGet, completed.Artifacts.KeypointsURL, nil))
	if keypointsRecorder.Code != http.StatusOK {
		t.Fatalf("keypoints status = %d", keypointsRecorder.Code)
	}
	var keypoints map[string]any
	decodeJSON(t, keypointsRecorder, &keypoints)
	if _, ok := keypoints["frames"]; !ok {
		t.Fatalf("keypoints = %+v", keypoints)
	}
}

func TestFailedHandPoseJobReportsFailure(t *testing.T) {
	server := newTestServerWithHandPose(t, fakeTranscriber{}, fakeHandPose{err: context.DeadlineExceeded})
	request := uploadRequest(t, "hands.mp4", []byte("fake media"), "hand_pose")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	var created job
	decodeJSON(t, recorder, &created)
	completed := waitForJob(t, server, created.ID)
	if completed.Status != "failed" {
		t.Fatalf("status = %q, want failed", completed.Status)
	}
	if completed.Error == nil || completed.Error.Code != "hand_pose_failed" {
		t.Fatalf("error = %+v", completed.Error)
	}
}

func TestCreateJobRejectsUnsupportedMedia(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})
	request := uploadRequest(t, "notes.txt", []byte("not media"), "asr")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
	payload := decodeResponse(t, recorder)
	if payload.Error == nil || payload.Error.Code != "unsupported_media" {
		t.Fatalf("payload = %+v", payload)
	}
}

func TestUnknownJobReturnsNotFound(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})
	request := httptest.NewRequest(http.MethodGet, "/api/v1/jobs/job_missing", nil)
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNotFound)
	}
}

func TestUpgradeReturnsOfficialAction(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})
	request := httptest.NewRequest(http.MethodPost, "/api/v1/chatgpt-plus/upgrade", strings.NewReader(`{}`))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	payload := decodeResponse(t, recorder)
	if !payload.Success || payload.Status != "user_action_required" {
		t.Fatalf("payload = %+v", payload)
	}
	if payload.UpgradeURL != plusUpgradeURL {
		t.Fatalf("upgrade URL = %q, want %q", payload.UpgradeURL, plusUpgradeURL)
	}
}

func TestUpgradeRejectsCredentials(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})
	request := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/chatgpt-plus/upgrade",
		strings.NewReader(`{"account":"person@example.com","password":"secret"}`),
	)
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
	payload := decodeResponse(t, recorder)
	if payload.Success || payload.Error == nil || payload.Error.Code != "credentials_not_accepted" {
		t.Fatalf("payload = %+v", payload)
	}
}

func TestUpgradeRequiresJSON(t *testing.T) {
	server := newTestServer(t, fakeTranscriber{})
	request := httptest.NewRequest(http.MethodPost, "/api/v1/chatgpt-plus/upgrade", strings.NewReader(""))
	recorder := httptest.NewRecorder()

	server.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusUnsupportedMediaType {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusUnsupportedMediaType)
	}
}

func newTestServer(t *testing.T, transcriber Transcriber) http.Handler {
	t.Helper()
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return New(Config{
		AllowedOrigin:  "http://localhost:4173",
		DataRoot:       t.TempDir(),
		MaxUploadBytes: 1 << 20,
		Transcriber:    transcriber,
	}, logger)
}

func newTestServerWithHandPose(t *testing.T, transcriber Transcriber, handPose HandPoseEstimator) http.Handler {
	t.Helper()
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	return New(Config{
		AllowedOrigin:  "http://localhost:4173",
		DataRoot:       t.TempDir(),
		MaxUploadBytes: 1 << 20,
		Transcriber:    transcriber,
		HandPose:       handPose,
	}, logger)
}

func uploadRequest(t *testing.T, filename string, content []byte, operator string) *http.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	if err := writer.WriteField("operator", operator); err != nil {
		t.Fatal(err)
	}
	part, err := writer.CreateFormFile("file", filename)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(content); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/api/v1/jobs", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	return request
}

func waitForJob(t *testing.T, server http.Handler, id string) job {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		request := httptest.NewRequest(http.MethodGet, "/api/v1/jobs/"+id, nil)
		recorder := httptest.NewRecorder()
		server.ServeHTTP(recorder, request)
		var item job
		decodeJSON(t, recorder, &item)
		if item.Status == "succeeded" || item.Status == "failed" {
			return item
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("job did not finish")
	return job{}
}

func decodeResponse(t *testing.T, recorder *httptest.ResponseRecorder) response {
	t.Helper()
	var payload response
	decodeJSON(t, recorder, &payload)
	return payload
}

func decodeJSON(t *testing.T, recorder *httptest.ResponseRecorder, target any) {
	t.Helper()
	if err := json.NewDecoder(recorder.Body).Decode(target); err != nil {
		t.Fatalf("decode response: %v", err)
	}
}

func TestSavedUploadUsesJobDirectory(t *testing.T) {
	root := t.TempDir()
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	server := New(Config{
		AllowedOrigin:  "http://localhost:4173",
		DataRoot:       root,
		MaxUploadBytes: 1 << 20,
		Transcriber:    fakeTranscriber{},
	}, logger)
	request := uploadRequest(t, "../../unsafe.mp4", []byte("fake media"), "asr")
	recorder := httptest.NewRecorder()
	server.ServeHTTP(recorder, request)
	var created job
	decodeJSON(t, recorder, &created)

	matches, err := filepath.Glob(filepath.Join(root, "jobs", created.ID, "input", "source.mp4"))
	if err != nil || len(matches) != 1 {
		t.Fatalf("saved upload matches = %v, error = %v", matches, err)
	}
}
