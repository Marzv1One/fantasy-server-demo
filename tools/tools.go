package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"charm.land/fantasy"
)

// NotifyFunc is called to send a JSON-RPC notification to the client.
type NotifyFunc func(method string, params interface{}) error

// RegisterAll registers all built-in tools on the given agent options slice
// and returns the updated slice.
func RegisterAll(notify NotifyFunc) []fantasy.AgentTool {
	return []fantasy.AgentTool{
		FileRead(notify),
		FileWrite(notify),
		Shell(notify),
	}
}

type fileReadInput struct {
	Path string `json:"path" description:"Absolute or relative file path to read"`
}

// FileRead returns a tool that reads a file and returns its contents.
func FileRead(notify NotifyFunc) fantasy.AgentTool {
	return fantasy.NewAgentTool("file_read", "Read a file and return its contents", func(ctx context.Context, input fileReadInput, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
		data, err := os.ReadFile(input.Path)
		if err != nil {
			return fantasy.NewTextErrorResponse(fmt.Sprintf("read error: %s", err)), nil
		}
		return fantasy.NewTextResponse(string(data)), nil
	})
}

type fileWriteInput struct {
	Path    string `json:"path" description:"File path to write to"`
	Content string `json:"content" description:"Content to write"`
}

// FileWrite returns a tool that writes content to a file.
// Sends a notification before and after the write so the editor can react.
func FileWrite(notify NotifyFunc) fantasy.AgentTool {
	return fantasy.NewAgentTool("file_write", "Write content to a file, creating directories as needed", func(ctx context.Context, input fileWriteInput, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
		abs, _ := filepath.Abs(input.Path)

		// Notify editor before write.
		notify("file.willChange", map[string]any{
			"path": abs,
		})

		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			return fantasy.NewTextErrorResponse(fmt.Sprintf("mkdir error: %s", err)), nil
		}

		if err := os.WriteFile(abs, []byte(input.Content), 0o644); err != nil {
			return fantasy.NewTextErrorResponse(fmt.Sprintf("write error: %s", err)), nil
		}

		// Notify editor after write.
		notify("file.changed", map[string]any{
			"path":    abs,
			"content": input.Content,
		})

		return fantasy.NewTextResponse(fmt.Sprintf("wrote %s", abs)), nil
	})
}

type shellInput struct {
	Command string `json:"command" description:"Shell command to execute"`
	Dir     string `json:"dir,omitempty" description:"Working directory (optional)"`
}

// Shell returns a tool that runs a shell command.
func Shell(notify NotifyFunc) fantasy.AgentTool {
	return fantasy.NewAgentTool("shell", "Execute a shell command and return stdout+stderr", func(ctx context.Context, input shellInput, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
		notify("shell.executing", map[string]any{
			"command": input.Command,
		})

		cmd := exec.CommandContext(ctx, "sh", "-c", input.Command)
		if input.Dir != "" {
			cmd.Dir = input.Dir
		}

		out, err := cmd.CombinedOutput()
		output := strings.TrimSpace(string(out))

		if err != nil {
			return fantasy.NewTextResponse(fmt.Sprintf("exit error: %s\noutput:\n%s", err, output)), nil
		}

		notify("shell.done", map[string]any{
			"command": input.Command,
			"output":  output,
		})

		return fantasy.NewTextResponse(output), nil
	})
}

// ToolCallNotification is the payload sent when a tool is about to execute.
type ToolCallNotification struct {
	ToolName string          `json:"toolName"`
	Input    json.RawMessage `json:"input"`
}

// ToolResultNotification is the payload sent when a tool finishes.
type ToolResultNotification struct {
	ToolName string `json:"toolName"`
	Output   string `json:"output"`
	IsError  bool   `json:"isError"`
}
