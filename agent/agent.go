package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"

	"github.com/user/jsonrpc-server/db"
	"github.com/user/jsonrpc-server/jsonrpc"
	"github.com/user/jsonrpc-server/tools"
)

// Request is the payload for the chat.send JSON-RPC method.
type Request struct {
	Prompt    string     `json:"prompt"`
	SessionID string     `json:"sessionId,omitempty"`
	Model     string     `json:"model,omitempty"`
	Files     []FilePart `json:"files,omitempty"`
}

// FilePart is a base64-encoded file attachment.
type FilePart struct {
	Filename  string `json:"filename"`
	Data      []byte `json:"data"`
	MediaType string `json:"mediaType"`
}

const maxFileSize = 5 * 1024 * 1024 // 5MB

// Response is returned after the agent finishes.
type Response struct {
	Text      string `json:"text"`
	SessionID string `json:"sessionId"`
	Usage     Usage  `json:"usage"`
}

type Usage struct {
	InputTokens  int64 `json:"inputTokens"`
	OutputTokens int64 `json:"outputTokens"`
	TotalTokens  int64 `json:"totalTokens"`
}

// Agent wraps a fantasy agent and manages conversations.
type Agent struct {
	mu             sync.Mutex
	provider       fantasy.Provider
	database       *db.DB
	defaultModel   string
	smallModel     string
	multimodalModel string
	apiKey         string
	baseURL        string
	systemPrompt   string
	maxHistory     int
}

// Config holds the initial configuration for the agent.
type Config struct {
	APIKey         string
	BaseURL        string
	DefaultModel   string
	SmallModel     string
	MultimodalModel string
	SystemPrompt   string
	Database       *db.DB
	MaxHistory     int // max messages to send to LLM (0 = no limit)
}

// New creates a new Agent.
func New(cfg Config) (*Agent, error) {
	provider, err := openaicompat.New(
		openaicompat.WithBaseURL(cfg.BaseURL),
		openaicompat.WithAPIKey(cfg.APIKey),
	)
	if err != nil {
		return nil, fmt.Errorf("create provider: %w", err)
	}

	return &Agent{
		provider:        provider,
		database:        cfg.Database,
		defaultModel:    cfg.DefaultModel,
		smallModel:      cfg.SmallModel,
		multimodalModel: cfg.MultimodalModel,
		apiKey:          cfg.APIKey,
		baseURL:         cfg.BaseURL,
		systemPrompt:    cfg.SystemPrompt,
		maxHistory:      cfg.MaxHistory,
	}, nil
}

// SetDefaultModel changes the default model used when none is specified.
func (a *Agent) SetDefaultModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.defaultModel = model
}

// GetDefaultModel returns the current default model.
func (a *Agent) GetDefaultModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.defaultModel
}

// SetSmallModel changes the small/fast model.
func (a *Agent) SetSmallModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.smallModel = model
}

// GetSmallModel returns the current small model.
func (a *Agent) GetSmallModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.smallModel
}

// SetMultimodalModel changes the multimodal model.
func (a *Agent) SetMultimodalModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.multimodalModel = model
}

// GetMultimodalModel returns the current multimodal model.
func (a *Agent) GetMultimodalModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.multimodalModel
}

// GetModels returns the current model configuration.
func (a *Agent) GetModels() map[string]string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return map[string]string{
		"primary":    a.defaultModel,
		"small":      a.smallModel,
		"multimodal": a.multimodalModel,
	}
}

// SetSystemPrompt changes the system prompt.
func (a *Agent) SetSystemPrompt(prompt string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.systemPrompt = prompt
}

// GetDatabase returns the database instance.
func (a *Agent) GetDatabase() *db.DB {
	return a.database
}

func generateID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// ensureSession returns the session ID, creating a new session if needed.
func (a *Agent) ensureSession(sessionID, model string) (string, error) {
	if sessionID != "" {
		existing, err := a.database.GetSession(sessionID)
		if err != nil {
			return "", fmt.Errorf("check session: %w", err)
		}
		if existing != nil {
			return sessionID, nil
		}
	}

	// Create new session.
	if sessionID == "" {
		sessionID = generateID()
	}
	a.mu.Lock()
	sysPrompt := a.systemPrompt
	a.mu.Unlock()
	if _, err := a.database.CreateSession(sessionID, model, sysPrompt); err != nil {
		return "", fmt.Errorf("create session: %w", err)
	}
	return sessionID, nil
}

// buildHistory loads conversation history from DB and returns fantasy messages.
// If maxHistory is set, truncates older messages to stay within the limit.
func (a *Agent) buildHistory(sessionID string) ([]fantasy.Message, error) {
	msgs, err := a.database.GetMessages(sessionID)
	if err != nil {
		return nil, err
	}

	// Truncate if history exceeds limit.
	truncated := false
	if a.maxHistory > 0 && len(msgs) > a.maxHistory {
		msgs = msgs[len(msgs)-a.maxHistory:]
		truncated = true
	}

	var messages []fantasy.Message

	// Insert a note if we truncated, so the model knows context is partial.
	if truncated {
		messages = append(messages, fantasy.Message{
			Role:    fantasy.MessageRoleSystem,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Earlier conversation context has been truncated. The following messages are the most recent portion of an ongoing conversation."}},
		})
	}

	for _, m := range msgs {
		switch m.Role {
		case "user":
			messages = append(messages, fantasy.NewUserMessage(m.Content))
		case "assistant":
			messages = append(messages, fantasy.Message{
				Role:    fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{fantasy.TextPart{Text: m.Content}},
			})
		case "tool":
			messages = append(messages, fantasy.Message{
				Role:    fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{fantasy.TextPart{Text: m.Content}},
			})
		}
	}
	return messages, nil
}

// Run executes an agent request, streaming notifications via the JSON-RPC server.
func (a *Agent) Run(ctx context.Context, srv *jsonrpc.Server, req Request) (*Response, error) {
	a.mu.Lock()
	modelID := a.defaultModel
	multimodalID := a.multimodalModel
	systemPrompt := a.systemPrompt
	a.mu.Unlock()

	if req.Model != "" {
		modelID = req.Model
	} else if len(req.Files) > 0 && multimodalID != "" {
		// Auto-select multimodal model when files are attached.
		modelID = multimodalID
	}

	// Ensure session exists.
	sessionID, err := a.ensureSession(req.SessionID, modelID)
	if err != nil {
		return nil, err
	}

	// Save user message.
	if _, err := a.database.AddMessage(sessionID, "user", req.Prompt); err != nil {
		return nil, fmt.Errorf("save user message: %w", err)
	}

	// Load conversation history.
	history, err := a.buildHistory(sessionID)
	if err != nil {
		return nil, fmt.Errorf("load history: %w", err)
	}

	model, err := a.provider.LanguageModel(ctx, modelID)
	if err != nil {
		return nil, fmt.Errorf("model %q: %w", modelID, err)
	}

	notify := func(method string, params interface{}) error {
		return srv.Notify(method, params)
	}

	agentTools := tools.RegisterAll(notify)

	ag := fantasy.NewAgent(model,
		fantasy.WithSystemPrompt(systemPrompt),
		fantasy.WithTools(agentTools...),
		fantasy.WithStopConditions(fantasy.StepCountIs(20)),
	)

	var finalText string
	var totalUsage Usage

	// Convert request files to fantasy FileParts, validating size.
	var files []fantasy.FilePart
	for _, f := range req.Files {
		if len(f.Data) > maxFileSize {
			return nil, fmt.Errorf("file %q exceeds 5MB limit (%d bytes)", f.Filename, len(f.Data))
		}
		files = append(files, fantasy.FilePart{
			Filename:  f.Filename,
			Data:      f.Data,
			MediaType: f.MediaType,
		})
	}

	// When files are present, the fantasy framework requires Prompt to be set.
	// We also remove the last user message from history to avoid duplication,
	// since the framework will create a user message from Prompt + Files.
	streamPrompt := ""
	if len(files) > 0 {
		streamPrompt = req.Prompt
		if len(history) > 0 && history[len(history)-1].Role == fantasy.MessageRoleUser {
			history = history[:len(history)-1]
		}
	}

	streamCall := fantasy.AgentStreamCall{
		Prompt:   streamPrompt,
		Messages: history,
		Files:    files,

		OnTextDelta: func(id, text string) error {
			finalText += text
			return notify("chat.textDelta", map[string]any{
				"text": text,
			})
		},

		OnToolCall: func(tc fantasy.ToolCallContent) error {
			return notify("chat.toolCall", tools.ToolCallNotification{
				ToolName: tc.ToolName,
				Input:    json.RawMessage(tc.Input),
			})
		},

		OnToolResult: func(res fantasy.ToolResultContent) error {
			var output string
			var isError bool
			if text, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](res.Result); ok {
				output = text.Text
			} else if errRes, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](res.Result); ok {
				output = errRes.Error.Error()
				isError = true
			}
			return notify("chat.toolResult", tools.ToolResultNotification{
				ToolName: res.ToolName,
				Output:   output,
				IsError:  isError,
			})
		},

		OnStepFinish: func(step fantasy.StepResult) error {
			return notify("chat.stepFinish", map[string]any{
				"finishReason": step.FinishReason,
			})
		},

		OnFinish: func(result *fantasy.AgentResult) {
			totalUsage = Usage{
				InputTokens:  result.TotalUsage.InputTokens,
				OutputTokens: result.TotalUsage.OutputTokens,
				TotalTokens:  result.TotalUsage.TotalTokens,
			}
		},

		OnError: func(err error) {
			notify("chat.error", map[string]any{
				"error": err.Error(),
			})
		},
	}

	result, err := ag.Stream(ctx, streamCall)
	if err != nil {
		return nil, fmt.Errorf("agent stream: %w", err)
	}

	totalUsage = Usage{
		InputTokens:  result.TotalUsage.InputTokens,
		OutputTokens: result.TotalUsage.OutputTokens,
		TotalTokens:  result.TotalUsage.TotalTokens,
	}

	// Save assistant response.
	if finalText != "" {
		if _, err := a.database.AddMessage(sessionID, "assistant", finalText); err != nil {
			return nil, fmt.Errorf("save assistant message: %w", err)
		}
	}

	// Auto-title: use first 80 chars of first message if title is empty.
	count, _ := a.database.GetMessageCount(sessionID)
	if count <= 2 {
		title := strings.ReplaceAll(req.Prompt, "\n", " ")
		title = strings.ReplaceAll(title, "\r", "")
		if len(title) > 80 {
			title = title[:80] + "..."
		}
		a.database.UpdateSessionTitle(sessionID, title)
	}

	return &Response{
		Text:      finalText,
		SessionID: sessionID,
		Usage:     totalUsage,
	}, nil
}
