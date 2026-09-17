package asr

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type CommandRunner struct {
	Python     string
	Script     string
	WorkingDir string
	Model      string
	ModelCache string
}

func (r CommandRunner) Transcribe(ctx context.Context, inputPath, outputPath string) error {
	if err := requireFile(r.Python, "ASR Python runtime"); err != nil {
		return err
	}
	if err := requireFile(r.Script, "ASR operator"); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return fmt.Errorf("create ASR output directory: %w", err)
	}

	arguments := []string{r.Script, inputPath, "--output", outputPath}
	if r.Model != "" {
		arguments = append(arguments, "--model", r.Model)
	}
	if r.ModelCache != "" {
		arguments = append(arguments, "--model-cache", r.ModelCache)
	}
	command := exec.CommandContext(ctx, r.Python, arguments...)
	command.Dir = r.WorkingDir
	command.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	var output bytes.Buffer
	command.Stdout = &output
	command.Stderr = &output
	if err := command.Run(); err != nil {
		message := strings.TrimSpace(output.String())
		if len(message) > 2000 {
			message = message[len(message)-2000:]
		}
		return fmt.Errorf("run ASR operator: %w: %s", err, message)
	}
	return nil
}

func requireFile(path, label string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("%s is unavailable at %s: %w", label, path, err)
	}
	if info.IsDir() {
		return fmt.Errorf("%s points to a directory: %s", label, path)
	}
	return nil
}
