package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"charm.land/fantasy"
)

// NotifyFunc is called to send a JSON-RPC notification to the client.
type NotifyFunc func(method string, params interface{}) error

// ContextRequestFunc sends a JSON-RPC request with context support.
type ContextRequestFunc func(ctx context.Context, method string, params interface{}) (json.RawMessage, error)

// confirmationTimeout is the default timeout for waiting on user confirmations.
const confirmationTimeout = 2 * time.Minute

// RegisterAll registers all built-in tools on the given agent options slice
// and returns the updated slice.
func RegisterAll(notify NotifyFunc, request ContextRequestFunc) []fantasy.AgentTool {
	return []fantasy.AgentTool{
		FileRead(notify),
		FileWrite(notify, request),
		FileEdit(notify, request),
		Shell(notify, request),
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
// Reads the existing file, sends a confirmation request with old+new content,
// and only writes after the user accepts.
func FileWrite(notify NotifyFunc, request ContextRequestFunc) fantasy.AgentTool {
	return fantasy.NewAgentTool("file_write", "Write content to a file, creating directories as needed", func(ctx context.Context, input fileWriteInput, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
		abs, _ := filepath.Abs(input.Path)

		fmt.Fprintf(os.Stderr, "[file_write] path=%s content_len=%d\n", abs, len(input.Content))

		// Reject empty writes up front — avoids showing a misleading "delete everything" diff.
		if strings.TrimSpace(input.Content) == "" {
			return fantasy.NewTextErrorResponse("file_write rejected: content is empty"), nil
		}

		// Read existing content for diff.
		var oldContent string
		if data, err := os.ReadFile(abs); err == nil {
			oldContent = string(data)
		}

		// Short-circuit if content is identical.
		if oldContent == input.Content {
			return fantasy.NewTextResponse(fmt.Sprintf("%s already up to date", abs)), nil
		}

		// Send confirmation request to client with timeout.
		reqCtx, cancel := context.WithTimeout(ctx, confirmationTimeout)
		defer cancel()

		resp, err := request(reqCtx, "file.confirm", map[string]any{
			"id":          abs,
			"path":        abs,
			"oldContent":  oldContent,
			"newContent":  input.Content,
		})
		if err != nil {
			return fantasy.NewTextErrorResponse(fmt.Sprintf("write cancelled: %s", err)), nil
		}

		var result struct {
			Accepted bool `json:"accepted"`
		}
		if err := json.Unmarshal(resp, &result); err != nil || !result.Accepted {
			return fantasy.NewTextErrorResponse("write cancelled by user"), nil
		}

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

type fileEditEntry struct {
	OldString string `json:"old_string" description:"Exact text to find (must match exactly including whitespace)"`
	NewString string `json:"new_string" description:"Replacement text"`
}

type fileEditInput struct {
	Path  string          `json:"path" description:"File path to edit"`
	Edits []fileEditEntry `json:"edits" description:"List of find-replace edits to apply sequentially"`
}

// FileEdit returns a tool that applies targeted find/replace edits to a file.
// Only the specified sections are changed, preserving the rest of the file.
func FileEdit(notify NotifyFunc, request ContextRequestFunc) fantasy.AgentTool {
	return fantasy.NewAgentTool("file_edit", "Apply targeted edits to a file by replacing specific text sections. Use this for partial changes instead of rewriting the entire file.", func(ctx context.Context, input fileEditInput, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
		abs, _ := filepath.Abs(input.Path)

		if len(input.Edits) == 0 {
			return fantasy.NewTextErrorResponse("file_edit requires at least one edit"), nil
		}

		data, err := os.ReadFile(abs)
		if err != nil {
			return fantasy.NewTextErrorResponse(fmt.Sprintf("read error: %s", err)), nil
		}
		content := string(data)

		// Apply edits sequentially, validating each one.
		for i, ed := range input.Edits {
			if ed.OldString == ed.NewString {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("edit %d: old_string and new_string are identical", i)), nil
			}
			if !strings.Contains(content, ed.OldString) {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("edit %d: old_string not found in file", i)), nil
			}
			// Ensure the old_string is unique to avoid ambiguous replacements.
			if strings.Count(content, ed.OldString) > 1 {
				return fantasy.NewTextErrorResponse(fmt.Sprintf("edit %d: old_string matches %d locations — provide more context to make it unique", i, strings.Count(content, ed.OldString))), nil
			}
			content = strings.Replace(content, ed.OldString, ed.NewString, 1)
		}

		// Short-circuit if nothing changed.
		if content == string(data) {
			return fantasy.NewTextResponse(fmt.Sprintf("%s already up to date", abs)), nil
		}

		// Send confirmation request to client with timeout.
		reqCtx, cancel := context.WithTimeout(ctx, confirmationTimeout)
		defer cancel()

		resp, err := request(reqCtx, "file.confirm", map[string]any{
			"id":          abs,
			"path":        abs,
			"oldContent":  string(data),
			"newContent":  content,
		})
		if err != nil {
			return fantasy.NewTextErrorResponse(fmt.Sprintf("edit cancelled: %s", err)), nil
		}

		var result struct {
			Accepted bool `json:"accepted"`
		}
		if err := json.Unmarshal(resp, &result); err != nil || !result.Accepted {
			return fantasy.NewTextErrorResponse("edit cancelled by user"), nil
		}

		// Notify editor before write.
		notify("file.willChange", map[string]any{
			"path": abs,
		})

		if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
			return fantasy.NewTextErrorResponse(fmt.Sprintf("write error: %s", err)), nil
		}

		// Notify editor after write.
		notify("file.changed", map[string]any{
			"path":    abs,
			"content": content,
		})

		return fantasy.NewTextResponse(fmt.Sprintf("edited %s (%d changes)", abs, len(input.Edits))), nil
	})
}

type shellInput struct {
	Command string `json:"command" description:"Shell command to execute"`
	Dir     string `json:"dir,omitempty" description:"Working directory (optional)"`
}

// Shell returns a tool that runs a shell command.
// Sends a confirmation request before execution.
func Shell(notify NotifyFunc, request ContextRequestFunc) fantasy.AgentTool {
	return fantasy.NewAgentTool("shell", "Execute a shell command and return stdout+stderr", func(ctx context.Context, input shellInput, call fantasy.ToolCall) (fantasy.ToolResponse, error) {
		// Send confirmation request to client with timeout.
		reqCtx, cancel := context.WithTimeout(ctx, confirmationTimeout)
		defer cancel()

		resp, err := request(reqCtx, "shell.confirm", map[string]any{
			"command": input.Command,
			"dir":     input.Dir,
		})
		if err != nil {
			return fantasy.NewTextErrorResponse(fmt.Sprintf("shell cancelled: %s", err)), nil
		}

		var result struct {
			Accepted bool `json:"accepted"`
		}
		if err := json.Unmarshal(resp, &result); err != nil || !result.Accepted {
			return fantasy.NewTextErrorResponse("shell cancelled by user"), nil
		}

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
