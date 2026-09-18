# DemoAI Server

Backend services for the DemoAI website.

## Catalog APIs

The web frontend renders no product or operator copy of its own: it fetches the
navigation and operator catalogs from this service, so adding, renaming, or
hiding an entry is a backend-only change.

```http
GET /api/v1/catalog/products   # navigation product menu
GET /api/v1/operators          # operator marketplace cards
GET /api/v1/operators/{id}     # single operator definition
```

Products carry `id`, `name`, `description`, `url`, and `enabled`. A product is
only linked when it is both enabled and has a URL; the `operator-marketplace`
entry is what exposes the 算子广场 page. Operators carry their availability,
accepted extensions, upload limit, and pricing, so the frontend never hard-codes
upload rules.

## ASR playground

The website can upload a video or audio file and start an asynchronous ASR job.
The server stores the job under `~/Desktop/DemoAI-TrainingData/jobs`, then calls
the existing `DemoAI-data/operators/asr/transcribe_video.py` operator. Only the
`asr` operator is accepted; clients cannot provide commands or filesystem paths.

### Run locally

From the `DemoAI-server` repository:

```bash
go run ./cmd/server
```

The server listens on port `8080` by default. Configuration:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PORT` | `8080` | HTTP listening port |
| `ALLOWED_ORIGIN` | `http://localhost:4173` | Frontend CORS origin |
| `DATA_ROOT` | `~/Desktop/DemoAI-TrainingData` | Uploaded files, models, and results |
| `DEMOAI_DATA_REPO` | `../DemoAI-data` | Local data-operator repository |
| `ASR_PYTHON` | `../DemoAI-data/.venv312/bin/python` | Python runtime containing MLX Whisper |
| `ASR_MODEL` | `mlx-community/whisper-small-mlx` | Whisper model name or local path |
| `ASR_MODEL_CACHE` | `$DATA_ROOT/models/asr` | Model cache directory |
| `MAX_UPLOAD_BYTES` | `524288000` | Maximum upload size |

### API

Create a job with multipart form data:

```http
POST /api/v1/jobs
Content-Type: multipart/form-data

operator=asr
file=@sample.mp4
```

The response is HTTP `202` with a job ID. Poll its status:

```http
GET /api/v1/jobs/{job_id}
```

After the job reaches `succeeded`, download the complete ASR JSON:

```http
GET /api/v1/jobs/{job_id}/artifacts/transcript
```

Jobs are stored on disk, while job state is intentionally in memory for this
local demo. Restarting the server clears the API's job index but does not delete
uploaded files or results.

## ChatGPT Plus upgrade flow

OpenAI does not provide a supported public API for purchasing a personal
ChatGPT Plus subscription with an account password. This service never accepts,
stores, or forwards ChatGPT credentials or payment details. It returns the
official Plus upgrade page so the account owner can sign in and complete the
purchase directly with OpenAI.

### Health check

```http
GET /healthz
```

### Request an official upgrade link

```http
POST /api/v1/chatgpt-plus/upgrade
Content-Type: application/json

{}
```

Response:

```json
{
  "success": true,
  "status": "user_action_required",
  "message": "Open the official upgrade page and complete sign-in and payment directly with OpenAI.",
  "upgrade_url": "https://chatgpt.com/plans/plus/"
}
```

If a caller sends an `account`, `email`, or `password` field, the service returns
HTTP `400` with `credentials_not_accepted`. Request bodies are never logged.

## Test

```bash
go test ./...
```

Official OpenAI API authentication uses API keys rather than ChatGPT account
passwords: https://platform.openai.com/docs/api-reference/introduction
