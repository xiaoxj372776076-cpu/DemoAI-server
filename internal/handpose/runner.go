package handpose

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// CommandRunner shells out to the DemoAI-data hand pose operator. The
// operator reads the video with OpenCV and encodes the result with a
// static ffmpeg, so BinDir is prepended to PATH.
type CommandRunner struct {
	Python     string
	Script     string
	WorkingDir string
	HaworRoot  string
	BinDir     string
}

// Estimate runs the operator over inputPath and returns the two artifacts it
// writes: the rendered video and the per-frame keypoint document. Both are
// derived from the input stem, which the server fixes to "source".
func (r CommandRunner) Estimate(ctx context.Context, inputPath, outputDirectory string) (string, string, error) {
	if err := requireFile(r.Python, "hand pose Python runtime"); err != nil {
		return "", "", err
	}
	if err := requireFile(r.Script, "hand pose operator"); err != nil {
		return "", "", err
	}
	if err := os.MkdirAll(outputDirectory, 0o755); err != nil {
		return "", "", fmt.Errorf("create hand pose output directory: %w", err)
	}

	arguments := []string{r.Script, inputPath, "--output-dir", outputDirectory}
	if r.HaworRoot != "" {
		arguments = append(arguments, "--hawor-root", r.HaworRoot)
	}
	command := exec.CommandContext(ctx, r.Python, arguments...)
	command.Dir = r.WorkingDir
	command.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	if r.BinDir != "" {
		command.Env = append(command.Env, "PATH="+r.BinDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	}
	var output bytes.Buffer
	command.Stdout = &output
	command.Stderr = &output
	runErr := command.Run()

	// The operator names both artifacts after the input stem, which the
	// server always fixes to "source".
	stem := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))
	videoPath := filepath.Join(outputDirectory, stem+"_hand_pose.mp4")
	keypointsPath := filepath.Join(outputDirectory, stem+"_hand_keypoints.json")

	// The operator writes both artifacts before its best-effort cleanup, so a
	// non-zero exit with complete output is still a usable result.
	videoErr := requireFile(videoPath, "hand pose rendered video")
	keypointsErr := requireFile(keypointsPath, "hand pose keypoints")
	if videoErr == nil && keypointsErr == nil {
		return videoPath, keypointsPath, nil
	}
	if runErr != nil {
		message := strings.TrimSpace(output.String())
		if len(message) > 2000 {
			message = message[len(message)-2000:]
		}
		return "", "", fmt.Errorf("run hand pose operator: %w: %s", runErr, message)
	}
	if videoErr != nil {
		return "", "", videoErr
	}
	return "", "", keypointsErr
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
